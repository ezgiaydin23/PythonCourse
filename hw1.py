import random

class Portfolio():
    def __init__(self):
        self.cash = 0.0
        self.stocks = {}
        self.mutualfunds = {}
        self.history = []

#show history of portfolio        
    def showHistory(self):
        print("----------TRANSACTION HISTORY----------")
        for h in self.history:
            print("\n", h)
        print("---------------------------------------")
        
#show details of portfolio
    def showDetails(self):
        print("PORTFOLIO DETAILS")
        print(f"Cash : {self.cash}")
        print(f"Stocks : {self.stocks}")
        print(f"Mutual Funds : {self.mutualfunds}")
    
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

#buy and sell stock functions

    def buyStock(self, share, Stock):
        
        if self.cash >= Stock.price * share :
            self.cash -= Stock.price * share 
            
            if Stock.symbol in self.stocks:
                self.stocks[Stock.symbol] += share
                
            else:
                self.stocks[Stock.symbol]= share
            
            self.history.append(f'Transaction Type : Buy Stock : {Stock.symbol} - Number of Shares : {share} Total Balance : {self.cash} $')                    
            print("You have purchased : ", share, Stock.symbol)     
        else:
            print("You do not have enough cash to buy : ", share, Stock.symbol)

    def sellStock(self, share, Stock):
        if self.stocks[Stock.symbol] >= share:
            self.stocks[Stock.symbol] -= share
            self.cash += random.uniform(0.5, 1.5) * Stock.price * share
            self.history.append(f'Transaction Type : Sell Stock : {Stock.symbol} - Number of Shares : {share} Total Balance : {self.cash} $')   
            print("You have sold : ", share, Stock.symbol) 
        else:
            print("You do not have enough stock to sell : ", share, Stock.symbol)
            
#buy and sell mutual funds functions

    def buyMutualFunds(self, amount, MutualFund):
        
        if self.cash >= amount:
            self.cash -= amount 
            
            if MutualFund.symbol in self.mutualfunds:
                self.mutualfunds[MutualFund.symbol] += amount
                
            else:
                self.mutualfunds[MutualFund.symbol] = amount
            
            self.history.append(f'Transaction Type : Buy Mutual Fund : {MutualFund.symbol} - Amount : {amount} Total Balance : {self.cash} $')                                        
            print("You have purchased : ", amount, MutualFund.symbol)     
        else:
            print("You do not have enough cash to buy : ", amount, MutualFund.symbol)

    def sellMutualFunds(self, amount, MutualFund):
        if self.mutualfunds[MutualFund.symbol] >= amount:
            self.mutualfunds[MutualFund.symbol] -= amount
            self.cash += random.uniform(0.9, 1.2) * amount
            self.history.append(f'Transaction Type : Sell Mutual Fund : {MutualFund.symbol} - Amount : {amount} Total Balance : {self.cash} $')              
            print("You have sold : ", amount, MutualFund.symbol)
        else:
            print("You do not have enough mutual fund to sell : ", amount, MutualFund.symbol)
            
class Stock():
     def __init__(self, price, symbol):
        self.price = price
        self.symbol = symbol
        
        
class MutualFund():
    def __init__(self, symbol):
        self.symbol = symbol
        self.price = 1.0
        
        
#test
p = Portfolio()
p.addCash(500)
p.addCash(100)
p.withdrawCash(100)

s=Stock(5,"EZG")
p.buyStock(5, s)
p.sellStock(5, s)

s2 = Stock(7, "RCP")
p.buyStock(1, s2)

m = MutualFund("MFA")
p.buyMutualFunds(5.5, m)
p.sellMutualFunds(3, m)

m2 = MutualFund("MFB")
p.buyMutualFunds(1, m2)

p.showHistory()
p.showDetails()