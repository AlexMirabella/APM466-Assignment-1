# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows,
# actions, and settings.

import datetime

import matplotlib.pyplot as plt
import numpy as np
import yaml

DATES = ['1/8/2024', '1/9/2024', '1/10/2024', '1/11/2024', '1/12/2024',
         '1/15/2024', '1/16/2024', '1/17/2024', '1/18/2024', '1/19/2024']


def term(j: int, d: datetime.datetime) -> float:
    return (d - datetime.datetime.strptime(DATES[j], '%m/%d/%Y')).days / 365


class Bond:
    isin: str
    issue_date: str
    maturity_date: str
    coupon: float
    prices: list[float]

    def __init__(self, isin: str, issue_date: str, maturity_date: str,
                 coupon: float):
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


def load_data() -> dict:
    parser: dict = yaml.safe_load(open('data.yaml', 'r'))
    return parser


def load_bonds(bond_data: dict) -> list[Bond]:
    bonds = []

    for isin in bond_data.keys():
        issue_date = bond_data[isin][0]
        maturity_date = bond_data[isin][1]
        coupon = bond_data[isin][2]
        bond = Bond(isin, issue_date, maturity_date, coupon)
        bond.prices = [price for price in bond_data[isin][3]]
        bonds.append(bond)

    return bonds


def date(d: str) -> datetime.datetime:
    return datetime.datetime.strptime(d, '%m/%d/%Y')


def plot_yield_curve(bonds: list[Bond]):
    maturities = [b.maturity() for b in bonds]

    for i in range(10):
        terms = [(maturity - datetime.datetime.strptime(DATES[i],
                                                        '%m/%d/%Y')).days / 365
                 for maturity in maturities]
        yields = [ytm(bonds, j, i) for j, b in enumerate(bonds)]
        # plt.plot(term, yields, color=(1 - i/10, 0.5-i/20, 0.1 + i/10))
        plt.plot(terms, yields, color='black')

    plt.title('Yield Curve', fontsize=18)
    plt.xlabel('Term (Years)', fontsize=15)
    plt.ylabel('Yield-To-Maturity (%)', fontsize=15)
    plt.xticks(fontsize=10)
    plt.tight_layout()
    plt.show()


def ytm(bonds: list[Bond], i: int, j: int) -> float:
    """

    :param bonds: list of bonds
    :param i: the index of the bond
    :param j: the index for the day of data collection
    :return: the yield-to-maturity of that bond on that day
    """
    bond = bonds[i]
    price = bond.dirty_price(j)
    coupon_pmt = bond.coupon / 2
    # Now we solve a polynomial equation for the yield-to-maturity
    # I made the variable r=(1 + ytm/2) for ease of calculation
    coeff = [price]
    for k in range(i):
        coeff.append(-coupon_pmt)
    coeff.append(-coupon_pmt - 100)
    rts = np.roots(coeff)
    # Find the root that's in the correct range from 1 to 2
    r = 0
    for x in rts:
        if np.imag(x) == 0.0 and 1 <= x <= 2:
            r = np.real(x)
    # ytm is calculated from value of r
    y = 2 * (r - 1)
    # yield is returned as a percentage
    return 100 * y


def yield_log_return(bonds: list[Bond]):
    ylr = [[] for _ in range(5)]
    for i in range(5):
        for j in range(9):
            ylr[i].append(np.log(ytm(bonds, i, j + 1) / ytm(bonds, i, j)))
    return ylr


def spot_yield(bonds: list[Bond]) -> list[list[float]]:
    """

    :param bonds:
    :return: a list of lists of spot yields for each day,
    i.e. first index is time, second is term
    """
    spot_rates = [[] for _ in range(10)]
    for i, bond in enumerate(bonds):
        coupon_pmt = bond.coupon / 2
        # Use previously calculated spot rates for each day
        # to calculate discounted cash flows
        for j in range(10):
            dirty_price = bond.dirty_price(j)
            discount = coupon_pmt * \
                       sum([np.exp(-rate * term(j, bonds[k].maturity())) for
                            k, rate in enumerate(spot_rates[j])])
            residual = dirty_price - discount
            # Handle cases where residual is too low
            if residual <= 0:
                print("Warning: the residual for bond" + str(i + 1)
                      + "for day " + DATES[j] + "is negative. "
                                                "Adjusting calculation.")
                spot_rates[j].append(spot_rates[j][i - 1])
            else:
                spot_rates[j].append(np.log((coupon_pmt + 100) / residual)
                                     / term(j, bond.maturity()))
    return spot_rates


def plot_spot_curve(bonds: list[Bond]):
    spot_rates = spot_yield(bonds)

    for j in range(10):
        terms = [term(j, bonds[i].maturity()) for i in range(10)]
        spot_rates_per_day = [100 * spot_rates[j][i] for i in range(10)]
        plt.plot(terms, spot_rates_per_day,
                 color=(1 - j / 10, 0.5 - j / 20, 0.1 + j / 10))
        # plt.plot(terms, spot_rates_per_day, color='black')

    plt.title('Spot Curve', fontsize=18)
    plt.xlabel('Term (Years)', fontsize=15)
    plt.ylabel('Spot Rate (%)', fontsize=15)
    plt.xticks(fontsize=10)
    plt.tight_layout()
    plt.show()


def forward_rates(bonds: list[Bond]):
    spot_rates = spot_yield(bonds)
    forward_rates = [[] for _ in range(10)]
    for j in range(10):
        for i in range(8):
            forward_rates[j].append(
                (spot_rates[j][i + 2] * term(j, bonds[i + 2].maturity())
                 - spot_rates[j][1] * term(j, bonds[1].maturity()))
                / ((bonds[i + 1].maturity() - bonds[1].maturity()).days / 365))
    return forward_rates


def forward_log_return(bonds: list[Bond]):
    forwards = forward_rates(bonds)
    flr = [[] for _ in range(4)]
    for i in range(4):
        for j in range(9):
            flr[i].append(
                np.log(forwards[j + 1][2 * i + 1] / forwards[j][2 * i + 1]))
    return flr


def plot_forward_curve(bonds: list[Bond]):
    rates = forward_rates(bonds)

    for j in range(10):
        terms = ["1yr", "2yr", "3yr", "4yr"]
        forward_rates_per_day = [100 * rates[j][2 * i + 1] for i in range(4)]
        # plt.plot(term, yields, color=(1 - i/10, 0.5-i/20, 0.1 + i/10))
        plt.plot(terms, forward_rates_per_day, color='black')

    plt.title('One Year Forward Curve', fontsize=18)
    plt.xlabel('Term', fontsize=15)
    plt.ylabel('Forward Rate (%)', fontsize=15)
    plt.xticks(fontsize=10)
    plt.tight_layout()
    plt.show()


def covariance_matrix(time_series: list[list[float]]) -> np.array:
    return np.cov(time_series)


if __name__ == '__main__':

    # Loading in the bond data from data.yaml
    bond_data = load_data()
    bonds = load_bonds(bond_data)

    # Making sure the bonds are sorted from earliest to latest maturity
    bonds.sort(key=lambda x: x.maturity())

    # --- Curve Plots ---
    # plot_yield_curve(bonds)
    # plot_spot_curve(bonds)
    # plot_forward_curve(bonds)

    # --- Covariance Matrices ---
    yield_log_covariance = covariance_matrix(yield_log_return(bonds))
    forward_log_covariance = covariance_matrix(forward_log_return(bonds))

    # print(yield_log_covariance)
    # print(forward_log_covariance)

    # --- Eigenvalues and Eigenvectors ---
    print(np.linalg.eig(yield_log_covariance))
    print(np.linalg.eig(forward_log_covariance))
