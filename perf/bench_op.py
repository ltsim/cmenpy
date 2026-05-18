import typing

import pyperf


def create_teardown():
    return """
def samelist(list1, list2):
    if len(list1) != len(list2):
        return False

    sorted_list1 = sorted(list1)
    sorted_list2 = sorted(list2)

    return sorted_list1 == sorted_list2


def issorted(arr):
    n = len(arr)

    for i in range(1, n):
        if arr[i - 1] > arr[i]:
            return False

    return True


assert samelist(sorted_elements, elements), "The list is no longer the same"
assert issorted(sorted_elements), "Sorting failed"
"""


def create_stmt(sort_name: str, stmt: str) -> str:
    return f"""
{stmt}

sorted_elements = {sort_name}(elements)
    """


def create_custom_stmt(sort_name: str, stmt: str, teardown_stmt: str) -> str:
    return f"""
{stmt}

{teardown_stmt}
    """


def run_all_bench(runner: pyperf.Runner, test: typing.Dict[str, str], setup: str, teardown: str):
    for name, stmt in test.items():
        try:
            runner.timeit(name,
                          stmt=stmt,
                          setup=setup,
                          teardown=teardown)
        except:
            continue

    return runner


if __name__ == "__main__":
    runner = pyperf.Runner()
