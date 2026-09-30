"""Architecture-neutral physical KV-cache projection tests."""

from __future__ import annotations

import unittest

import torch

from vosti_kernels import physical


class WeightTensorMechanicsTests(unittest.TestCase):
    def test_collection_preserves_layer_role_order_and_object_identity(self):
        tensors = [torch.ones(2) for _ in range(4)]
        named = physical.named_layer_tensors(
            [tuple(tensors[:2]), tuple(tensors[2:])], ("norm", "projection"), label="weights"
        )
        self.assertEqual([name for name, _ in named], [
            "layer 0 norm", "layer 0 projection", "layer 1 norm", "layer 1 projection"
        ])
        self.assertTrue(all(actual is expected for (_, actual), expected in zip(named, tensors)))
        self.assertEqual(physical.named_layer_tensors([], ("norm",), label="weights"), [])

    def test_collection_rejects_truncation_and_non_tuple_layers(self):
        tensor = torch.ones(2)
        for layer in ((), (tensor,), (tensor, tensor, tensor), [tensor, tensor], tensor):
            with self.subTest(layer=type(layer)), self.assertRaisesRegex(
                RuntimeError, "weights layer 0 is not an exact role tuple"
            ):
                physical.named_layer_tensors([layer], ("norm", "projection"), label="weights")

    def test_collection_does_not_replace_permission_checks(self):
        for tensor, message in (
            (None, "not a tensor"),
            (torch.ones(2, 3).T, "not contiguous"),
            (torch.ones(2, requires_grad=True), "requires gradients"),
        ):
            named = physical.named_layer_tensors([(tensor,)], ("norm",), label="weights")
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                physical.validate_named_tensors(named, label="weights")

    def test_shapes_are_exact_and_support_per_layer_geometry(self):
        for width in (2, 4):
            expected = {"norm": (width,), "projection": (width, 8)}
            tensors = {role: torch.empty(shape) for role, shape in expected.items()}
            physical.validate_tensor_shapes(tensors.items(), expected, label="layer")
            for role, tensor in tensors.items():
                # Same element count is insufficient: ranks and dimensions matter.
                with self.subTest(width=width, role=role), self.assertRaisesRegex(
                    RuntimeError, f"layer {role} has shape"
                ):
                    physical.validate_tensor_shapes(
                        [(role, tensor.reshape(1, -1))], expected, label="layer"
                    )
        with self.assertRaises(KeyError):
            physical.validate_tensor_shapes([("unknown", torch.ones(2))], {}, label="layer")


class ModelKVCacheAllocationTests(unittest.TestCase):
    def test_layer_projection_validator_rejects_shape_and_alias_drift(self) -> None:
        embed = torch.zeros((7, 4), dtype=torch.bfloat16)
        projections = (torch.zeros((8, 4), dtype=torch.bfloat16),
                       torch.zeros((8, 4), dtype=torch.bfloat16))
        dimensions = (4, 8)
        caches = physical.init_layer_model_kv_caches(embed, projections, dimensions, 65)
        physical.validate_layer_model_kv_caches(caches, embed, projections, dimensions, 65)
        for changed, message in (
            (list(reversed(caches)), "expected"),
            ([(caches[0][0], caches[0][0]), caches[1]], "same tensor object"),
            ([caches[0], (caches[0][0].view(2, 64, 1, 8), caches[1][1])], "shared storage"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                physical.validate_layer_model_kv_caches(changed, embed, projections, dimensions, 65)

    def test_heterogeneous_layers_share_page_indices_not_tensor_shapes(self) -> None:
        tails = ((2, 4), (1, 8), (2, 4))
        kwargs = dict(num_layers=3, token_capacity=65, layer_tail_shapes=tails,
                      dtype=torch.bfloat16, device="cpu")
        caches = physical.allocate_kv_cache_collection(**kwargs, zero_initialize=True)
        physical.validate_kv_cache_collection(caches, **kwargs, label="heterogeneous")
        for pair, tail in zip(caches, tails):
            for tensor in pair:
                self.assertEqual(tuple(tensor.shape), (2, physical.PAGE_SIZE, *tail))
                self.assertEqual(tensor.count_nonzero().item(), 0)
        with self.assertRaisesRegex(RuntimeError, "shared storage"):
            physical.validate_kv_cache_collection(
                [caches[0], caches[1], (caches[0][0].view_as(caches[0][0]), caches[2][1])],
                **kwargs, label="heterogeneous")
        with self.assertRaisesRegex(RuntimeError, "expected"):
            physical.validate_kv_cache_collection(
                [caches[1], caches[0], caches[2]], **kwargs, label="heterogeneous")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            physical.allocate_kv_cache_collection(**kwargs, tail_shape=(2, 4))
        with self.assertRaisesRegex(ValueError, "every layer"):
            physical.allocate_kv_cache_collection(**{**kwargs, "layer_tail_shapes": tails[:2]})
        with self.assertRaisesRegex(ValueError, "positive integers"):
            physical.allocate_kv_cache_collection(**{**kwargs, "layer_tail_shapes": ((0, 4),) * 3})

    def test_derives_geometry_from_weights_and_sealed_head_dimension(self) -> None:
        for dtype, kv_width, kv_heads in (
            (torch.bfloat16, 6, 3), (torch.float32, 4, 2)
        ):
            with self.subTest(dtype=dtype, kv_width=kv_width):
                embed = torch.zeros((11, 4), dtype=dtype)
                k_proj = torch.zeros((kv_width, 4), dtype=dtype)
                caches = physical.init_model_kv_caches(
                    embed, k_proj, 2, 2, physical.PAGE_SIZE + 1
                )
                tensors = [tensor for pair in caches for tensor in pair]
                expected = (2, physical.PAGE_SIZE, kv_heads, 2)
                self.assertTrue(all(tuple(tensor.shape) == expected for tensor in tensors))
                self.assertEqual(
                    {(tensor.dtype, tensor.device) for tensor in tensors},
                    {(dtype, embed.device)},
                )
                self.assertEqual(len({id(tensor) for tensor in tensors}), len(tensors))
                self.assertEqual(
                    len({tensor.untyped_storage().data_ptr() for tensor in tensors}),
                    len(tensors),
                )

    def test_rejects_geometry_dtype_or_alias_drift(self) -> None:
        embed = torch.zeros((11, 4), dtype=torch.bfloat16)
        k_proj = torch.zeros((6, 4), dtype=torch.bfloat16)
        valid = physical.init_model_kv_caches(
            embed, k_proj, 2, 1, physical.PAGE_SIZE
        )
        with self.assertRaisesRegex(RuntimeError, "same tensor object"):
            physical.validate_model_kv_caches(
                [(valid[0][0], valid[0][0])], embed, k_proj, 2,
                1, physical.PAGE_SIZE,
            )
        with self.assertRaisesRegex(RuntimeError, "inconsistent"):
            physical.validate_model_kv_caches(
                valid, embed, torch.zeros((5, 4), dtype=torch.bfloat16), 2,
                1, physical.PAGE_SIZE,
            )
        with self.assertRaisesRegex(RuntimeError, "one dtype and device"):
            physical.validate_model_kv_caches(
                valid, embed, k_proj.float(), 2, 1, physical.PAGE_SIZE,
            )

        with self.assertRaisesRegex(RuntimeError, "head dimension"):
            physical.init_model_kv_caches(embed, k_proj, 0, 1, physical.PAGE_SIZE)


if __name__ == "__main__":
    unittest.main()
