import yaml

import main

if __name__=='__main__':
    d: dict = yaml.safe_load(open('data.yaml', 'r'))
    bonds = []
    new = {}
    for isin in d.keys():
        issue_date = d[isin][0]
        maturity_date = d[isin][1]
        coupon = d[isin][2]
        bond = main.Bond(isin, issue_date, maturity_date, coupon)
        bond.prices = [price for price in d[isin][3]]
        new.update(bond.to_dict())
    yaml.dump(new, open('readable_data.yaml', 'w'))


