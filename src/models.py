class Expense:
    def __init__(self, name: str, amount: float, paid_by: int, split_between: list[int]) -> None:
        assert amount > 0
        self.name = name
        self.amount = float(amount)
        self.paid_by = paid_by
        self.split_between = split_between

    def __str__(self):
        return f"Expense '{self.name}': {self.paid_by} --{self.amount}--> {self.split_between}"


class Transfer:
    def __init__(self, name: str, amount: float, paid_by: int, paid_to: int) -> None:
        assert amount > 0
        self.name = name
        self.amount = amount
        self.paid_by = paid_by
        self.paid_to = paid_to

    def __str__(self):
        return f"Transfer '{self.name}': {self.paid_by} --{self.amount}-> {self.paid_to}"
