"""Conservative definition provenance over the existing typed statement tree.

IDs are temporary analysis keys, never serialized IR. Zero denotes an input
or possibly undefined value. Loops use a finite may-definition fixed point,
including the zero-iteration path; branches join both predecessor sets.
"""

from . import Assign, For, If, Kernel, Let, MaskedStore, TensorView, Var


def assigned_variables(statements) -> frozenset[str]:
    """All possibly written names, including nested and partial writes.

    A parametric forward loop cannot retain entry/exit facts for these values
    without an invariant. This is an effect inventory, not definite assignment
    or evidence that any particular loop iteration executes.
    """
    names = set()
    for statement in statements:
        match statement:
            case Assign(target=Var(name=name)) | Let(var=Var(name=name)):
                names.add(name)
            case Assign(target=TensorView(base=Var(name=name))) | MaskedStore(base=Var(name=name)):
                names.add(name)
            case If(then_body=then_body, else_body=else_body):
                names.update(assigned_variables((*then_body, *else_body)))
            case For(var=Var(name=name), body=body):
                names.add(name)
                names.update(assigned_variables(body))
            case _:
                raise ValueError(f"unsupported definition effect statement: {statement!r}")
    return frozenset(names)


def reaching_definitions(kernel: Kernel) -> dict[int, dict[str, frozenset[int]]]:
    before: dict[int, dict[str, frozenset[int]]] = {}

    def join(left, right):
        return {name: left.get(name, frozenset({0})) | right.get(name, frozenset({0}))
                for name in left.keys() | right.keys()}

    def walk(statements, incoming):
        state = dict(incoming)
        for statement in statements:
            key = id(statement)
            before[key] = join(before[key], state) if key in before else dict(state)
            match statement:
                case Assign(target=Var(name=name)) | Let(var=Var(name=name)):
                    state[name] = frozenset({key})
                case Assign(target=TensorView(base=Var(name=name))) | MaskedStore(base=Var(name=name)):
                    state[name] = state.get(name, frozenset({0})) | {key}
                case If(then_body=then_body, else_body=else_body):
                    state = join(walk(then_body, state), walk(else_body, state))
                case For(var=Var(name=name), body=body):
                    entry = dict(state)
                    header = dict(entry)
                    while True:
                        iteration = {**header, name: frozenset({key})}
                        updated = join(entry, walk(body, iteration))
                        if updated == header:
                            break
                        header = updated
                    state = header
                case _:
                    raise ValueError(f"unsupported definition provenance statement: {statement!r}")
        return state

    walk(kernel.grid.body, {parameter.name: frozenset({0}) for parameter in kernel.params})
    return before
