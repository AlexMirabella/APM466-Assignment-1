# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows,
# actions, and settings.

import yaml
import datetime


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

    def save(self, parser: dict):
        if self.isin in parser.keys():
            if input("Bond already exists. Replace?") != 'yes':
                return None

        parser[self.isin] = [self.issue_date, self.maturity_date,
                             self.coupon, self.prices]

    def to_dict(self):
        return {self.isin: {"Coupon": self.coupon,
                            "Maturity Date": self.maturity_date,
                            "Issue Date": self.issue_date,
                            "Prices": [{DATES[i]: self.prices[i]}
                                       for i in range(self.prices.__len__())]}}

    def maturity(self):
        return date(self.maturity_date)

    def __str__(self):
        return 'CAN ' + str(self.coupon) + ' ' + \
            self.maturity().strftime('%b %y')


def load_data():
    parser: dict = yaml.safe_load(open('data.yaml', 'r'))
    return parser


def save_data(parser: dict):
    yaml.dump(parser, open('data.yaml', 'w'))


def get_bond():
    isin = input("ISIN: ")
    issue_date = input("Issue Date: ")
    mat_date = input("Maturity Date: ")
    coupon = float(input("Coupon:"))
    bond = Bond(isin, issue_date, mat_date, coupon)
    print("Enter Prices:")
    get_prices(bond)
    return bond


def get_prices(bond: Bond):
    confirm = False
    while not confirm:
        p = input(DATES[bond.prices.__len__()] + ' -- ')
        if p == 'b' and bond.prices.__len__() != 0:
            bond.prices.pop()
        else:
            try:
                bond.prices.append(float(p))
            except ValueError:
                print('Error: invalid entry')

        while bond.prices.__len__() == 10 and not confirm:
            t = input('Confirm? ')
            if t == 'b':
                bond.prices.pop()
            elif t == 'y':
                confirm = True


def date(d: str) -> datetime.datetime:
    return datetime.datetime.strptime(d, '%m/%d/%Y')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    bond_data = yaml.safe_load(open('data.yaml', 'r'))
    bonds = []

    for isin in bond_data.keys():
        issue_date = bond_data[isin][0]
        maturity_date = bond_data[isin][1]
        coupon = bond_data[isin][2]
        bond = Bond(isin, issue_date, maturity_date, coupon)
        bond.prices = [price for price in bond_data[isin][3]]
        bonds.append(bond)

    for bond in bonds:
        print(bond)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
