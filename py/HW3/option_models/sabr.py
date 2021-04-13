# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10

@author: jaehyuk
"""

import numpy as np
from . import normal
from . import bsm
import scipy.integrate as spint
import sys

sys.path.insert(sys.path.index('') + 1, 'C:/Users/cherr/Documents/GitHub/PyFeng')
import pyfeng as pf

'''
MC model class for Beta=1
'''


class ModelBsmMC:
    beta = 1.0  # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    bsm_model = None
    '''
    You may define more members for MC: time step, etc
    '''

    def __init__(self, sigma, vov=0, rho=0.0, beta=1.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.bsm_model = pf.Bsm(sigma, intr=intr, divr=divr)

    def bsm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of bsm_vol in ModelHagan class
        use bsm_model
        '''
        return 0

    def price(self, strike, spot, texp=None, cp=1):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        dt = 0.01
        n_time = int(texp // dt)
        n_path = 10000
        sigma_path = np.zeros((n_time+1, n_path))
        sigma_path[0, :] = self.sigma
        stock_path = np.zeros((n_time + 1, n_path))
        stock_path[0, :] = spot
        for i in range(n_time):
            z1 = np.random.standard_normal(n_path)
            x1 = np.random.standard_normal(n_path)
            w1 = self.rho * z1 + np.sqrt(1 - self.rho ** 2) * x1
            sigma_path[i+1, :] = sigma_path[i, :] * np.exp(self.vov * np.sqrt(dt) * z1 - 0.5 * self.vov ** 2 * dt)
            stock_path[i+1, :] = stock_path[i, :] * np.exp(sigma_path[i, :] * np.sqrt(dt) * w1 - 0.5 * sigma_path[i, :] ** 2 * dt)
        S_T = stock_path[-1, :]
        price = []
        std = []
        for k in strike:
            p = np.mean(np.fmax(S_T-k, 0))
            price.append(p)
            std.append(np.std(np.fmax(S_T-k, 0)))
        # print("The standard deviation: \n", np.array(std))
        return np.array(price)

    def price0(self, strike, spot, texp=None, cp=1):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''

        m = pf.BsmNdMc(self.sigma, rn_seed=12345)
        m.simulate(n_path=20000, tobs=[texp])
        p = []
        payoff = lambda x: np.fmax(np.mean(x, axis=1) - s, 0)  # Basket option
        for s in strike:
            p.append(m.price_european(spot, texp, payoff))
        return np.array(p)


'''
MC model class for Beta=0
'''


class ModelNormalMC:
    beta = 0.0  # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    normal_model = None

    def __init__(self, sigma, vov=0, rho=0.0, beta=0.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.normal_model = pf.Norm(sigma, intr=intr, divr=divr)

    def norm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of normal_vol in ModelNormalHagan class
        use normal_model 
        '''
        return 0

    def price(self, strike, spot, texp=None, sigma=None, cp=1):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        dt = 0.01
        n_time = int(texp // dt)
        n_path = 10000
        sigma_path = np.zeros((n_time + 1, n_path))
        sigma_path[0, :] = self.sigma
        stock_path = np.zeros((n_time + 1, n_path))
        stock_path[0, :] = spot
        for i in range(n_time):
            z1 = np.random.standard_normal(n_path)
            x1 = np.random.standard_normal(n_path)
            w1 = self.rho * z1 + np.sqrt(1 - self.rho ** 2) * x1
            sigma_path[i + 1, :] = sigma_path[i, :] * np.exp(self.vov * np.sqrt(dt) * z1 - 0.5 * self.vov ** 2 * dt)
            stock_path[i + 1, :] = stock_path[i, :] + sigma_path[i, :] * w1 * np.sqrt(dt)
        S_T = stock_path[-1, :]
        price = []
        std = []
        for k in strike:
            p = np.mean(np.fmax(S_T - k, 0))
            price.append(p)
            std.append(np.std(np.fmax(S_T - k, 0)))
        # print("The standard deviation: \n", np.array(std))

        return np.array(price)

    def price0(self, strike, spot, texp=None, sigma=None, cp=1):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        m = pf.NormNdMc(self.sigma, rn_seed=12345)
        m.simulate(n_path=20000, tobs=[texp])
        p = []
        payoff = lambda x: np.fmax(np.mean(x, axis=1) - s, 0)  # Basket option
        for s in strike:
            p.append(m.price_european(spot, texp, payoff))
        return np.array(p)


'''
Conditional MC model class for Beta=1
'''


class ModelBsmCondMC:
    beta = 1.0  # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    bsm_model = None
    '''
    You may define more members for MC: time step, etc
    '''

    def __init__(self, sigma, vov=0, rho=0.0, beta=1.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.bsm_model = pf.Bsm(sigma, intr=intr, divr=divr)

    def bsm_vol(self, strike, spot, texp=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of bsm_vol in ModelHagan class
        use bsm_model
        should be same as bsm_vol method in ModelBsmMC (just copy & paste)
        '''
        return 0

    def price(self, strike, spot, texp=None, cp=1):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and normal price.
        You may fix the random number seed
        '''
        # method 1
        dt = 0.01
        n_time = int(texp // dt)
        n_path = 10000
        sigma_path = np.zeros((n_time + 1, n_path))
        sigma_path[0, :] = self.sigma
        for i in range(n_time):
            z1 = np.random.standard_normal(n_path)
            sigma_path[i + 1, :] = sigma_path[i, :] * np.exp(self.vov * np.sqrt(dt) * z1 - 0.5 * self.vov ** 2 * dt)
        sigma0 = sigma_path[0, :]
        sigma_final = sigma_path[-1, :]
        int_var = np.mean(sigma_path ** 2, axis=0)

        spot_mc = spot * np.exp(self.rho * (sigma_final - sigma0) / self.vov - 0.5 * self.rho ** 2 * int_var)
        vol = np.sqrt((1 - self.rho ** 2) * int_var / texp)

        price = []
        for s in strike:
            price.append(bsm.price(s, spot_mc, texp, vol, cp_sign=cp))

        # print("The standard deviation: \n", np.std(price, axis=1))

        return np.mean(price, axis=1)

    def price0(self, strike, spot, texp=None, cp=1):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and BSM price.
        Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        # method 2
        m = pf.BsmNdMc(self.vov, rn_seed=12345)
        tobs = np.arange(0, 101) / 100 * texp
        _ = m.simulate(tobs=tobs, n_path=1000)
        sigma_path = np.squeeze(m.path) * self.sigma
        sigma0 = sigma_path[0, :]
        sigma_final = sigma_path[-1, :]

        int_var = spint.simps(sigma_path ** 2, dx=1, axis=0) / 100

        spot_mc = spot * np.exp(self.rho * (sigma_final - sigma0) / self.vov - 0.5 * self.rho ** 2 * int_var)
        vol = np.sqrt((1 - self.rho ** 2) * int_var / texp)

        price = []
        for s in strike:
            price.append(bsm.price(s, spot_mc, texp, vol, cp_sign=cp))

        return np.mean(price, axis=1)



'''
Conditional MC model class for Beta=0
'''


class ModelNormalCondMC:
    beta = 0.0  # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    normal_model = None

    def __init__(self, sigma, vov=0, rho=0.0, beta=0.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.normal_model = pf.Norm(sigma, intr=intr, divr=divr)

    def norm_vol(self, strike, spot, texp=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of normal_vol in ModelNormalHagan class
        use normal_model
        should be same as norm_vol method in ModelNormalMC (just copy & paste)
        '''
            
        return 0

    def price(self, strike, spot, texp=None, cp=1):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and normal price.
        You may fix the random number seed
        '''
        # method 1
        dt = 0.01
        n_time = int(texp // dt)
        n_path = 10000
        sigma_path = np.zeros((n_time + 1, n_path))
        sigma_path[0, :] = self.sigma
        for i in range(n_time):
            z1 = np.random.standard_normal(n_path)
            sigma_path[i + 1, :] = sigma_path[i, :] * np.exp(self.vov * np.sqrt(dt) * z1 - 0.5 * self.vov ** 2 * dt)
        sigma0 = sigma_path[0, :]
        sigma_final = sigma_path[-1, :]
        int_var = np.mean(sigma_path ** 2, axis=0)
        # int_var = spint.simps(sigma_path ** 2, dx=1, axis=0) / 100

        spot_mc = spot + self.rho * (sigma_final - sigma0) / self.vov
        vol = np.sqrt((1 - self.rho ** 2) * int_var / texp)

        price = []
        for s in strike:
            price.append(normal.price(s, spot_mc, texp, vol, cp_sign=cp))
        # print("The standard deviation: \n", np.std(price, axis=1))

        return np.mean(price, axis=1)

    def price0(self, strike, spot, texp=None, cp=1):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and normal price.
        You may fix the random number seed
        '''
        # method 2
        m = pf.NormNdMc(self.vov, rn_seed=12345)
        tobs = np.arange(0, 101) / 100 * texp
        _ = m.simulate(tobs=tobs, n_path=1000)
        sigma_path = np.squeeze(m.path) * self.sigma
        sigma0 = sigma_path[0, :]
        sigma_final = sigma_path[-1, :]

        int_var = spint.simps(sigma_path ** 2, dx=1, axis=0) / 100

        spot_mc = spot + self.rho * (sigma_final - sigma0) / self.vov
        vol = np.sqrt((1 - self.rho ** 2) * int_var / texp)

        price = []
        for s in strike:
            price.append(normal.price(s, spot_mc, texp, vol, cp_sign=cp))

        return np.mean(price, axis=1)
