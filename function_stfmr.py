# coding: utf-8
import numpy as np

class fit_func_stfmr():
    def __init__(self):
        super().__init__()
        self.mu_B = 9.23e-24
        self.hbar = 1.05e-34
        self.e = 1.6021e-19
        self.mu = 4*np.pi*1e-7
        self.planck =6.62607*1e-34

    # definition basical function
    def linear(self,x,a,b):
        return a*x+b

    def parabolic(self,x,a,b,c):
        return a*x**2+b*x+c

    #　definition fitting function for AMR
    def AMR(self,theta,R,dR,b,c):
        return R-dR*(np.cos((theta-b)/180*np.pi))**2+c*theta


    # definition fitting function for JH
    def DC(self,I,a,b,c):
        return a*pow(I,2)+b*I+c

    def RF(self,P,a,b):
        return a*P+b


    #関数の定義
    #フィッティング関数
    #対称・反対称ローレンツ関数
    def f3(self, x, Vsym, dH,HFMR,Vasy):
        return Vsym*(dH*dH)/(pow((x-HFMR),2)+dH*dH)+Vasy*(dH*(x-HFMR))/(pow((x-HFMR),2)+dH*dH)
    def STFMR(self, x, Vsym, dH,HFMR,Vasy,c,d):
        return Vsym*(dH*dH)/(pow((x-HFMR),2)+dH*dH)+Vasy*(dH*(x-HFMR))/(pow((x-HFMR),2)+dH*dH)+c*x+d
    def f1(self, x, Vsym, dH,HFMR):
        return Vsym*(dH*dH)/(pow((x-HFMR),2)+dH*dH)
    def f0(self, x,  dH,HFMR,Vasy):
        return Vasy*(dH*(x-HFMR))/(pow((x-HFMR),2)+dH*dH)


    #キッテルの式(ガンマ固定)
    def kittel_fixed(self,x, Ms,Hu,g=2.099):
        gamma=g*self.mu_B/self.hbar*1e-12
        return gamma/2/np.pi*pow((x+Hu)*(x+Ms+Hu),0.5)


    #線幅周波数依存性
    def linewidth(self,x, a, H0):
        return H0+a*2*np.pi*x


    # effective field

    # H_DL
    def H_DL(self,S,Hr,W,Meff,DR,I_RF):
        Asym=W*(2*Hr+Meff)/(Hr+Meff)*pow(1+Meff/Hr,0.5)
        HDL=2*pow(2,0.5)*Asym*S*1e-6/DR/(I_RF*1e-3)
        return HDL

    # H_|| (H_Oe+H_FL) and H_FL and H_Oe
    def H_parallel(self,A,Hr,W,Meff,DR,I_RF):
        Aasy=W*(2*Hr+Meff)/(Hr+Meff)
        H_parallel=2*pow(2,0.5)*Aasy*A*1e-6/DR/(I_RF*1e-3)
        return H_parallel


    # Torque effieciency

    # ξ_FMR
    def xi_FMR1(self,An,Ap,Sn,Sp,Ms,d_F,d_N,Meff,HFMR):
        xi_fmr = (abs(Sp-Sn))/(abs(An)+abs(Ap))*self.e/self.hbar*Ms*d_F*d_N*pow(1+Meff/HFMR,0.5)
        return xi_fmr
    def xi_FMR2(self,A,S,Ms,d_F,d_N,Meff,HFMR):
        xi_fmr = S/A*self.e/self.hbar*Ms*d_F*d_N*pow(1+Meff/HFMR,0.5)
        return xi_fmr

    # ξ_eff^E
    def xi_E(self,H_eff,E,d_F,Ms):
        zeta = H_eff*1e-3/E
        xi_eff_E = 2*self.e/self.hbar*H_eff*1e-3/E*d_F*Ms/self.mu/100
        return xi_eff_E, zeta

class gene_calc:
    # Coefficient of determination
    def R2(self,data,ideal):
        data = np.array(data)
        ideal = np.array(ideal)
        residuals = data -ideal
        rss = np.sum(residuals**2)
        p = np.mean(data)
        tss = np.sum((data-p)**2)
        r2 = 1-(rss/tss)
        return r2

    # Error propagation
    #Reciprocal　1/(M1±e1)
    def inver(self,M1,e1):
        return 1/(M1**2)*e1
    #sum and difference　(M1±e1)±(M2±e2)
    def sum_dif(self,e1,e2):
        return pow(e1**2+e2**2,0.5)
    #product　(M1±e1)×(M2±e2)
    def product(self,M1,M2,e1,e2):
        return pow((M2*e1)**2+(M1*e2)**2,0.5)
    #quotient　(M1±e1)/(M2±e2)
    def div(self,M1,M2,e1,e2):
        return pow((e1/M2)**2+(M1*e2/(M2**2))**2,0.5)
    #exponentiation　(M1±e1)^(M2±e2)
    def power_e(self,M1,M2,e1,e2):
        return pow((M2*M1**(M2-1)*e1)**2+(np.log(M1)*((M1)**M2)*e2)**2,0.5)