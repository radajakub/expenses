import numpy as np
from heapq import heapify, heappop, heappush
from timeit import default_timer as timer

from models import Expense, Transfer


class BalanceSheet:
    def __init__(self, users: list[int], expenses: list[Expense], transfers: list[Transfer]) -> None:
        self.expenses = expenses
        self.transfers = transfers
        self.users = users

        self.sum_trip_costs()
        self.compute_balances()

    def summary(self) -> None:
        print('-- Balance sheet --')
        print(f'Users: {self.users}')
        print(f'Total expenses: {self.trip_sum}')
        print('Balances:')
        for user, debt in self.balances.items():
            if debt > 0:
                print(f'User {user} should get {debt} currency')
            elif debt < 0:
                print(f'User {user} should pay {-debt} currency')

    def sum_trip_costs(self) -> None:
        self.trip_sum = np.sum(
            np.array([expense.amount for expense in self.expenses]))

    def compute_balances(self) -> None:
        self.balances = {key: value for key, value in zip(
            self.users, [0] * len(self.users))}
        for expense in self.expenses:
            self.balances[expense.paid_by] += expense.amount
            partial = expense.amount / len(expense.split_between)
            for user in expense.split_between:
                self.balances[user] -= partial
        for transfer in self.transfers:
            self.balances[transfer.paid_by] += transfer.amount
            self.balances[transfer.paid_to] -= transfer.amount

    def split(self, sign=1) -> tuple[list[tuple[float, int, int]], list[tuple[float, int]]]:
        positives = [(sign * debt, idx, user)
                     for idx, (user, debt) in enumerate(self.balances.items()) if debt > 0]
        negatives = [(sign * -debt, idx, user)
                     for idx, (user, debt) in enumerate(self.balances.items()) if debt < 0]
        return positives, negatives

    # naive settle up
    def naive_settle(self) -> None:
        settles = []
        positives, negatives = self.split()
        negatives.sort(reverse=True)
        positives.sort(reverse=True)
        while len(negatives) and len(positives):
            debt, neg_idx, debtor = negatives[0]
            credit, pos_idx, creditor = positives[0]
            if debt < credit:
                settles.append(Transfer('SETTLE', debt, debtor, creditor))
                negatives.pop(0)
                positives[0] = (credit - debt, pos_idx, creditor)
                positives.sort(reverse=True)
            elif debt > credit:
                settles.append(Transfer('SETTLE', credit, debtor, creditor))
                positives.pop(0)
                negatives[0] = (debt - credit, neg_idx, debtor)
                negatives.sort(reverse=True)
            else:
                settles.append(
                    Transfer('SETTLE', credit, debtor, creditor))
                positives.pop(0)
                negatives.pop(0)
        return settles

    def settle(self) -> list[Transfer]:
        settles = []
        positives, negatives = self.split(sign=-1)
        heapify(positives)
        heapify(negatives)
        while len(negatives) and len(positives):
            credit, pos_idx, creditor = heappop(positives)
            debt, neg_idx, debtor = heappop(negatives)
            debt = -debt
            credit = -credit
            settles.append(
                Transfer('SETTLE', min(debt, credit), debtor, creditor))
            diff = debt - credit
            if diff > 0:
                heappush(negatives, (-diff, neg_idx, debtor))
            elif diff < 0:
                heappush(positives, (diff, pos_idx, creditor))
        return settles


def benchmark(b: BalanceSheet) -> None:
    start = timer()
    _ = b.settle()
    t1 = timer() - start
    start = timer()
    _ = b.naive_settle()
    t2 = timer() - start
    return t2, t1


def sample_balance_sheet(users_count=10, lb=10, ub=10000, max_count=10000, rng=np.random.default_rng) -> BalanceSheet:
    users = [x for x in range(1, users_count + 1)]
    expenses_count = rng.integers(1, max_count, endpoint=True)
    expenses = []
    for i in range(expenses_count):
        amount = rng.integers(lb, ub, endpoint=True)
        creditor = rng.choice(users)
        debtor_count = rng.integers(1, users_count, endpoint=True)
        debtors = rng.choice(users, size=debtor_count, replace=False)
        expenses.append(
            Expense(f'E{i}', amount=amount, paid_by=creditor, split_between=debtors))
    return BalanceSheet(users, expenses, [])


def get_residuals(bs: BalanceSheet, settles: list[Transfer]) -> BalanceSheet:
    balances = bs.balances.copy()
    for settle in settles:
        balances[settle.paid_by] += settle.amount
        balances[settle.paid_to] -= settle.amount
    return [x for x in balances.values() if x != 0]


def compare(settle1, settle2) -> bool:
    for s1, s2 in zip(settle1, settle2):
        if s1.paid_by != s2.paid_by or s1.paid_to != s2.paid_to or s1.amount != s2.amount:
            return False
    return True


def grid_search(rng, user_min=1, user_max=4, expense_min=1, expense_max=4, trials=10):
    times = np.zeros(shape=(user_max-user_min+1,
                     expense_max-expense_min+1, 2, trials))
    ui = 0
    for user_power in range(user_min, user_max+1):
        user_count = 10**user_power
        ei = 0
        for expense_power in range(expense_min, expense_max+1):
            expense_count = 10**expense_power
            for i in range(trials):
                bs = sample_balance_sheet(
                    users_count=user_count, max_count=expense_count, rng=rng)
                naive_t, opt_t = benchmark(bs)
                times[ui, ei, 0, i] = naive_t
                times[ui, ei, 1, i] = opt_t
            ei += 1
        ui += 1
    times = np.mean(times, axis=3)

    print('speedup of optimized approach over naive approach')
    width = 10
    print(" " * (width + 2), end='|')
    for ei in range(times.shape[1]):
        print(
            f' {" " * (width - 1 - (expense_min + ei))}{10**(expense_min + ei)} ', end='|')
    print()
    for ui in range(times.shape[0]):
        print(
            f' {" " * (width - 1 - (user_min + ui))}{10**(user_min + ui)} ', end='|')
        for ei in range(times.shape[1]):
            print(
                f' {format(np.round(times[ui, ei, 0]/times[ui, ei, 1], decimals=width-2), f".{width-2}f")} ', end='|')
        print()


if __name__ == "__main__":
    rng = np.random.default_rng()
    users_count = 10
    lb = 10
    ub = 10000
    max_count = 10000
    balance_sheet = sample_balance_sheet(users_count=users_count, lb=lb, ub=ub, max_count=max_count, rng=rng)
    print('balance sheet sampled')
    balance_sheet.summary()

    print(f'naive approach')
    naive = balance_sheet.naive_settle()
    for i in naive:
        print(i)
    print(f'non-zero residuals: {get_residuals(balance_sheet, naive)}')
    print()

    print('optimized approach')
    opt = balance_sheet.settle()
    for i in opt:
        print(i)
    print(f'non-zero residuals: {get_residuals(balance_sheet, opt)}')
    print()

    print('parameter grid_search')
    grid_search(rng, user_min=1, user_max=3,
                expense_min=1, expense_max=5, trials=10)
