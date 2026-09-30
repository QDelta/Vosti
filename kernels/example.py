from ir.validate_subset import validate_triton_subset
from ir.translate import translate_kernel_source
from ir.pp import pretty_kernel
from ir.relational_verifier import verify_annotations

with open("triton_kernels/matmul.py") as f:
    source = f.read()

# Check the kernel uses only verifiable patterns (block_ptr, no raw pointers)
violations = validate_triton_subset(source, "matmul_kernel")
assert not violations, violations

# Translate Triton source to the verification IR
kernel = translate_kernel_source(source, "matmul_kernel")
print(pretty_kernel(kernel))

# Prove row equivalence using @pre/@post annotations
result = verify_annotations(
    source, "matmul_kernel", {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}
)
for check in result.checks:
    status = "PROVED" if check.proved else "FAILED"
    print(f"[{status}] {check.name}: {check.details}")
