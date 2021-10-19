import random

class Portfolio(object):
    def __init__(self):
        self.cash = 0.0
        self.assets={"Stock" : {}, "Mutual Fund" : {}, "Bonds": {}, "Crypto" : {}}
        self.history = []

#show history of portfolio        
    def showHistory(self):
        print("----------TRANSACTION HISTORY----------")
        for h in self.history:
            print("\n", h)
        print("---------------------------------------")
        
#show details of portfolio
    def __str__(self):
        details = f'PORTFOLIO DETAILS\nCash : {self.cash}'
        for asset in self.assets:
            details += f'\n{asset} : '
            if not self.assets[asset]: 
                details += '\t-\n'
            for a in self.assets[asset]:
                details += f'\n\t{a} : {self.assets[asset][a]}'
            
        return details
                
#add and withdraw cash functions  

    def addCash(self, cash):
        self.cash += cash
        self.history.append(f'Transaction Type : Add Cash : {cash} $ Total Balance : {self.cash} $')
        print("You have added : ", cash)    
        
    def withdrawCash(self, cash):
        if self.cash >= cash:
            self.cash -= cash
            self.history.append(f'Transaction Type : Withdraw Cash : {cash} $ Total Balance : {self.cash} $')
            print("You have withdraw : ", cash)  
        else:
            print("You do not have enough cash to withdraw : ", cash, "$")

#buy asset
    def buyAsset(self, share, Asset):
        if Asset.getType() == "Stock" : amount = int(share) 
        else : amount = share
        
        if self.cash >= Asset.price * amount:
            self.cash -= Asset.price * amount
            if Asset.symbol in self.assets[Asset.getType()]:
                self.assets[Asset.getType()][Asset.symbol] += amount
            else:
                self.assets[Asset.getType()][Asset.symbol] = amount
            print("You have purchased : ", amount, Asset.symbol)
            self.history.append(f'Transaction Type : Buy {Asset.getType()} : {Asset.symbol} - Number of Shares : {amount} Total Balance : {self.cash} $')                 
        else:
            print("You do not have enough cash to buy : ", amount, Asset.symbol)
            
            
#sell asset
    def sellAsset(self, share, Asset):
        if Asset.getType() == "Stock" : amount = int(share) 
        else : amount = share
        
        if Asset.symbol in self.assets[Asset.getType()]:
            if self.assets[Asset.getType()][Asset.symbol] >= amount:
                self.assets[Asset.getType()][Asset.symbol] -= amount
                self.cash += Asset.calcPrice() * amount
                print("You have sold : ", amount, Asset.symbol)
                self.history.append(f'Transaction Type : Sell {Asset.getType()} : {Asset.symbol} - Number of Shares : {amount} Total Balance : {self.cash} $')       
                if self.assets[Asset.getType()][Asset.symbol] == 0:
                    del self.assets[Asset.getType()][Asset.symbol]
            else:
                print("You do not have enough asset to sell : ", amount, Asset.symbol)
        else:
                print("You do not have enough asset to sell : ", amount, Asset.symbol)
  

#asset classes defined
            
class Asset():
    def __init__(self, price, symbol):
        self.price = price
        self.symbol = symbol
        
    def calcPrice(self):
        return random.uniform(0.9, 1.2) * self.price
            
class Stock(Asset):
    def __init__(self, price, symbol):
        Asset.__init__(self, price, symbol)
        
    def getType(self):
        return "Stock"
    
    def calcPrice(self):
        return random.uniform(0.5, 1.5) * self.price
        
class MutualFund(Asset):
    def __init__(self, symbol):
        Asset.__init__(self, 1.0, symbol)
        
    def getType(self):
        return "Mutual Fund"
    
class Bonds(Asset):
    def __init__(self, price, symbol):
        Asset.__init__(self, price, symbol)
        
    def getType(self):
        return "Bonds"
    
class Crypto(Asset):
    def __init__(self, price, symbol):
        Asset.__init__(self, price, symbol)
        
    def getType(self):
        return "Crypto"
    
    def calcPrice(self):
        return random.uniform(0.1, 3.0) * self.price
    
    
        
#test
s1 = Stock(50, "EVR")
s2 = Stock(10, "REZ")
mf1 = MutualFund("MFA")
b1 = Bonds(250, "BND")
c1 = Crypto(60, "BTC")

portfolio = Portfolio()
portfolio.addCash(1500)
portfolio.withdrawCash(500)


portfolio.buyAsset(5.5, s1)
portfolio.buyAsset(1, s2)
portfolio.buyAsset(100.5, mf1)
portfolio.buyAsset(2, b1)
portfolio.buyAsset(3, c1)

portfolio.sellAsset(4.5, s1)
portfolio.sellAsset(75, mf1)
portfolio.sellAsset(1, b1)
portfolio.sellAsset(1, c1)

portfolio.showHistory()
print(portfolio)