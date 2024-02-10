# This is a sample Python script.
import numpy
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows,
# actions, and settings.

import yaml
import datetime
from dateutil.relativedelta import relativedelta
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

DATES = ['1/8/2024', '1/9/2024', '1/10/2024', '1/11/2024', '1/12/2024',
         '1/15/2024', '1/16/2024', '1/17/2024', '1/18/2024', '1/19/2024']


class Bond:
    isin: str
    issue_date: str
    maturity_date: str
    coupon: float
    prices: list[float]

    def __init__(self, isin: str, issue_date: str, maturity_date: str, coupon: float):
        self.isin = isin
        self.issue_date = issue_date
        self.maturity_date = maturity_date
        self.coupon = coupon
        self.prices = []

    def add_price(self, price: float):
        self.prices.append(price)

    def to_dict(self):
        return {"ISIN": self.isin,
                "Coupon": self.coupon,
                "Maturity Date": self.maturity_date,
                "Issue Date": self.issue_date,
                "Prices": [{DATES[i]: self.prices[i]}
                           for i in range(self.prices.__len__())]}

    def maturity(self):
        return date(self.maturity_date)

    def __str__(self):
        return 'CAN ' + str(self.coupon) + ' ' + \
            self.maturity().strftime('%b %y')

    def dirty_price(self, t: int) -> float:
        """

        :param t: The day number for the data collected, from 0-9
        :return: the dirty price of the bond on that day
        """
        # All bonds were chosen to have the same
        # last coupon date of September 1st 2024
        last_coupon_date = datetime.datetime.strptime('9/1/23', '%m/%d/%y')
        today = datetime.datetime.strptime(DATES[t], '%m/%d/%Y')
        accrued_interest = self.coupon * ((today - last_coupon_date).days / 365)
        return self.prices[t] + accrued_interest


def load_data():
    parser: dict = yaml.safe_load(open('data.yaml', 'r'))
    return parser


def date(d: str) -> datetime.datetime:
    return datetime.datetime.strptime(d, '%m/%d/%Y')


def plot_yield_curve(bonds: list[Bond]):
    maturities = [b.maturity().strftime('%b \'%y') for b in bonds]
    for i in range(10):
        y = [ytm(b, i) for b in bonds]
        plt.plot(maturities, y, 'black')

    plt.title('Yield Curve', fontsize=18)
    plt.xlabel('Date', fontsize=15)
    plt.ylabel('Yield-To-Maturity (%)', fontsize=15)
    plt.xticks(rotation=30, fontsize=10)
    plt.tight_layout()
    plt.show()


def ytm(b: Bond, i: int) -> float:
    price = b.prices[i]
    coupon = b.coupon * 100 / 2
    y = (100 / price)**(1 / (1+i)) - 1
    return y


if __name__ == '__main__':

    # Loading in the bond data from data.yaml
    bond_data = yaml.safe_load(open('data.yaml', 'r'))
    bonds = []

    for isin in bond_data.keys():
        issue_date = bond_data[isin][0]
        maturity_date = bond_data[isin][1]
        coupon = bond_data[isin][2]
        bond = Bond(isin, issue_date, maturity_date, coupon)
        bond.prices = [price for price in bond_data[isin][3]]
        bonds.append(bond)

    # Making sure the bonds are sorted from earliest to latest maturity
    bonds.sort(key=lambda x: x.maturity())
    plot_yield_curve(bonds)




