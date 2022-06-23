# coding: utf-8

#fileの読み込み
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import codecs
import os
import matplotlib.cm as cm
from matplotlib import rc
#reference: http://oversleptabit.com/?p=556
import pandas as pd
import pathlib
# my module
from function_stfmr import gene_calc as calc
from function_stfmr import fit_func_stfmr as fit
from visualization import visualization as vis



class stfmr_parameter(fit,vis,calc):

    def __init__(self,g=2.099):
        # make result folder
        super().__init__()
        self.name1 = 'fit_parameter'
        self.name2 = 'fit_data'
        p1=pathlib.Path('./result/'+self.name1)
        p1.mkdir(exist_ok=True, parents = True)
        p2=pathlib.Path('./result/'+self.name2)
        p2.mkdir(exist_ok=True, parents = True)
        self.g = g
        self.gamma = self.g*self.mu_B/self.hbar*1e-12


    # fitting process(AMR)
    def singlefit_process_AMR(self,path_in,Plot=True):
        # load data file
        fp = codecs.open(path_in, encoding='cp932', errors='ignore')
        data = np.loadtxt(fp, comments='#', dtype='float', skiprows=0)

        p_sub = pathlib.Path(path_in) # path object  ディレクトリの階層を上がるため
        try:
            thickness = int(str(p_sub.parents[0])[-6:-4])
        except ValueError:
            thickness = 0
        x=data[:,0]
        y=data[:,1]
        ini=[100,10,10,0.001]
        params_bounds =[
            [0,-np.inf,-np.inf,-np.inf],
            [np.inf,np.inf,np.inf,np.inf]
        ]
        para_AMR,cov_AMR=curve_fit(fit().AMR, x ,y ,ini,bounds=params_bounds)
        angle=np.arange(0,360,0.5)
        if Plot:
            plt.plot(data[:,0],((data[:,1]-para_AMR[3]*data[:,0])/para_AMR[0]-1)*100,'o',label=str(thickness)+'nm')
            plt.plot(angle,((fit().AMR(angle,*para_AMR)-para_AMR[3]*angle)/para_AMR[0]-1)*100,'-')
            plt.xlabel(r'$\theta$ $\rm{(deg)}$')
            plt.ylabel(r'$\Delta R/R_{\perp}}~$ $(\%)$')
            plt.legend(loc=(1,0.3),ncol=2).get_frame().set_alpha(0)
            plt.xlim(0,360)
        #plt.ylim(-0.1,1.1)

        # fit data Dataframe
        Q = pd.DataFrame([[thickness]*len(angle),x,y,angle,fit().AMR(angle,*para_AMR)],\
            index = ['thickness(nm)','angle_raw(deg)','R_AMR_raw(ohm)','fit_angle(deg)','fit_AMR(ohm)'])
        Raw_d = Q.T

        # fit parameter DataFrame
        para_AMR = pd.DataFrame(para_AMR,index = ['R(ohm)','dR(ohm)','dtheta(deg)','a(ohm/deg)']).T
        sigma_AMR = np.sqrt(np.diag(cov_AMR))
        sigma_AMR = pd.DataFrame(sigma_AMR,index = ['R(ohm)_err','dR(ohm)_err','dtheta_err(deg)','a_err(ohm/deg)']).T
        thickness = pd.DataFrame([thickness],index = ['thickness(nm)']).T
        AMR_fit_para = pd.concat([thickness,para_AMR,sigma_AMR],axis=1)

        return Raw_d, AMR_fit_para

    def AMR_repeat_fit(self,path,txt,save=True,Plot=True):
        Raw_data = pd.DataFrame()
        Fit_para = pd.DataFrame()
        for pathname, dirnames, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith(txt):
                    path_in = os.path.join(pathname, filename)
                    if filename[0:3]=='AMR':
                        Raw_D, Fit_param = self.singlefit_process_AMR(path_in,Plot)
                        Raw_data = pd.concat([Raw_data,Raw_D],axis=0)
                        Fit_para = pd.concat([Fit_para,Fit_param],axis=0)
        Fit_para = Fit_para.sort_values('thickness(nm)')
        Fit_para = Fit_para.reset_index(drop=True)
        plt.show()
        if save:
            Raw_data.to_csv(f'./result/{self.name2}/AMR_fit_data.csv',index=False)
            Fit_para.to_csv(f'./result/{self.name1}/AMR_fit_parameter.csv',index=False)
        return Raw_data, Fit_para

    # fitting process Joule-heating
    # DC process
    def singlefit_process_DC(self,path_in,Plot=True):
        # load data file
        fp = codecs.open(path_in, encoding='cp932', errors='ignore')
        data = np.loadtxt(fp, comments='#', dtype='float', skiprows=0)

        p_sub = pathlib.Path(path_in) # path object  ディレクトリの階層を上がるため

        try:
            thickness = int(str(p_sub.parents[1])[-6:-4])
        except ValueError:
            thickness = 0
        x=data[:,0]*1e+3 #mA
        y=data[:,2]
        ini=[100,10,10]
        para_DC,cov_DC=curve_fit(fit().DC, x, y, ini)
        cur=np.arange(-max(x)-5,max(x)+5,0.25)
        if Plot:
            plt.plot(x,y-para_DC[2],'o',label=str(thickness)+'nm')
            plt.plot(cur,fit().DC(cur,*para_DC)-para_DC[2],'-')
            plt.xlabel(r'$I_{\rm{DC}}$ $\rm{(mA)}$')
            plt.ylabel(r'$\Delta R_{\rm{J}}~$ $(\Omega)$')
            plt.legend(loc=(1,0.0),ncol=2).get_frame().set_alpha(0)

        #fit data DataFrame
        Q = pd.DataFrame([[thickness]*len(cur),x,y,cur,fit().DC(cur,*para_DC)],\
            index = ['thickness(nm)','current_raw(mA)','R_DC_raw(ohm)','fit_current(mA)','fit_R_DC(ohm)'])
        Raw_d = Q.T
        # fit parameter DataFrame
        para_DC = pd.DataFrame(para_DC,index = ['a(ohm/(mA)^2)','b(ohm/mA)','R(ohm)']).T
        sigma_DC = np.sqrt(np.diag(cov_DC))
        sigma_DC = pd.DataFrame(sigma_DC,index = ['a(ohm/(mA)^2)_err','b(ohm/mA)_err','R_err(ohm)']).T
        thickness = pd.DataFrame([thickness],index = ['thickness(nm)']).T
        DC_fit_para = pd.concat([thickness,para_DC,sigma_DC],axis=1)
        return Raw_d,DC_fit_para


    def DC_repeat_fit(self,path,txt,name,save=True,Plot=True):
        Raw_data = pd.DataFrame()
        Fit_para = pd.DataFrame()
        for pathname, dirnames, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith(txt):
                    path_in = os.path.join(pathname, filename)
                    if filename[0:2]==name:
                        #print(int(pathname[-2:]),filename)
                        Raw_D,Fit_param=self.singlefit_process_DC(path_in,Plot)
                        Raw_data = pd.concat([Raw_data,Raw_D],axis=0)
                        Fit_para = pd.concat([Fit_para,Fit_param],axis=0)
        Fit_para = Fit_para.sort_values('thickness(nm)')
        Fit_para = Fit_para.reset_index(drop=True)
        plt.show()
        if save:
            Raw_data.to_csv(f'./result/{self.name2}/DC_fit_data.csv',index=False)
            Fit_para.to_csv(f'./result/{self.name1}/DC_fit_parameter.csv',index=False)
        return Raw_data, Fit_para

    # RF process
    def singlefit_process_RF(self,path_in,Plot=True):
        # load data file
        fp = codecs.open(path_in, encoding='cp932', errors='ignore')
        data = np.loadtxt(fp, comments='#', dtype='float', skiprows=0)

        p_sub = pathlib.Path(path_in) # path object # ディレクトリの階層を上がるため

        try:
            thickness = int(str(p_sub.parents[1])[-6:-4])
        except TypeError:
            thickness = 0
        NN = len(data[:,0])
        N = len(set(data[:,0])) # number of frequency
        n = NN//N # data points per freqency
        fit_data = pd.DataFrame()
        for i in range(N):
            x=data[n*i:n+n*i,1] # power (mW)
            y=data[n*i:n+n*i,4]*1e+3 # Resistnace(ohm)
            f=data[n*i,0] # frequency
            ini=[10,100]
            para_RF,cov_RF=curve_fit(fit().RF, x, y, ini)
            power=np.arange(0,max(x)+10,0.1)

            # visualization
            if Plot:
                plt.plot(x,y-para_RF[1],'o',label =str(thickness)+'nm'+str(f)+'GHz')
                plt.plot(power,fit().RF(power,*para_RF)-para_RF[1],'-')
                plt.xlabel(r'$P$ $\rm{(mW)}$')
                plt.ylabel(r'$\Delta R_{\rm{J}}~$ $(\Omega)$')
                plt.xlim(0,max(x)+10)
                plt.legend(loc=(1,0),ncol=4).get_frame().set_alpha(0)

            #fit data DataFrame
            Q = pd.DataFrame([[thickness]*len(power),x,y,power,fit().RF(power,*para_RF)],\
            index = ['thickness(nm)','RF_power_raw(mW)','R_EF_raw(ohm)','fit_RF_power(mW)','fit_R_RF(ohm)'])
            Raw_d = Q.T

            # fit parameter DataFrame
            para_RF = pd.DataFrame(para_RF,index = ['a(ohm/mW)','R(ohm)']).T
            sigma_RF = np.sqrt(np.diag(cov_RF))
            sigma_RF = pd.DataFrame(sigma_RF,index = ['a(ohm/mW)_err','R_err(ohm)']).T
            thickness1 = pd.DataFrame([thickness],index = ['thickness(nm)']).T
            frequency = pd.DataFrame([f],index =['frequency(GHz)']).T
            RF_fit_para = pd.concat([thickness1,frequency,para_RF,sigma_RF],axis=1)
            fit_data = pd.concat([fit_data,RF_fit_para],axis = 0)
        return Raw_d, fit_data

    def RF_repeat_fit(self,path,txt,name,save=True,Plot=True):
        Raw_data = pd.DataFrame()
        Fit_para = pd.DataFrame()
        for pathname, dirnames, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith(txt):
                    path_in = os.path.join(pathname, filename)
                    if filename[0:2]==name:
                        #print(int(pathname[-2:]),filename)
                        Raw_D,Fit_param=self.singlefit_process_RF(path_in,Plot)
                        Raw_data = pd.concat([Raw_data,Raw_D],axis=0)
                        Fit_para = pd.concat([Fit_para,Fit_param],axis=0)
        Fit_para = Fit_para.sort_values('thickness(nm)')
        Fit_para = Fit_para.reset_index(drop=True)
        plt.show()
        if save:
            Raw_data.to_csv(f'./result/{self.name2}/RF_fit_data.csv',index=False)
            Fit_para.to_csv(f'./result/{self.name1}/RF_fit_parameter.csv',index=False)
        return Raw_data, Fit_para

    # fitting process ST-FMR signal
    def singlefit_process_STFMR(self,path_in,Plot=True):
        # load data file
        fp = codecs.open(path_in, encoding='cp932', errors='ignore')
        data = np.loadtxt(fp, comments='#', dtype='float', skiprows=0)

        p_sub = pathlib.Path(path_in) # path object
        frequency=float(p_sub.name[10:14])

        try:
            thickness = int(str(p_sub.parents[1])[-6:-4])
        except ValueError:
            thickness = 0

        # parameter bounds (negative field)
        params_bounds_n =[
            [-np.inf,0,-np.inf,-np.inf,-np.inf,-np.inf],
            [np.inf,np.inf,0,np.inf,np.inf,np.inf]
        ]
        # fitting for nagtive field
        dN=len(data[:,0]) # data points
        xm=data[10:dN//2-20,0] # raw data for negative field
        ym=data[10:dN//2-20,1]*1e+6 # raw data for negative field
        cm = max(ym)
        c1m = np.argmax(ym)
        c2m = data[c1m,0]
        dm = min(ym)
        d1m = np.argmin(ym)
        d2m = data[d1m,0]
        # Symmetry, linewidth, resonance field, Antisym, slope, offset
        inim= [-10,20, (c2m+d2m)/2, np.absolute(cm-dm)*1.5, 0.0002, 1]  # initial parameter(nagative field)
        try:
            para_FMRm,cov_FMRm = curve_fit(fit().STFMR, xm, ym, inim, maxfev =12000, bounds=params_bounds_n)
        except RuntimeError:
            pass
        mag_m=np.arange(-300,0.1,0.1)

        #visualization of fitting result
        if Plot:
            fig, axes = plt.subplots(2, 1, sharex="col", tight_layout=False,figsize=(10,5))
            fig.subplots_adjust(wspace=0.1, hspace=0.1)
            axes[0].plot(xm,ym-para_FMRm[4]*xm-para_FMRm[5],'ok',markersize=3)
            axes[0].plot(mag_m,fit().STFMR(mag_m,*para_FMRm)-para_FMRm[4]*mag_m-para_FMRm[5],'-g')
            R2m = calc().R2(ym,fit().STFMR(xm,*para_FMRm))
            axes[1].plot(xm,ym-fit().STFMR(xm,*para_FMRm),'-g',label =f'R^2={R2m:.5f}')




        # dataframe for negative field
        #fit data DataFrame
        Q = pd.DataFrame([[thickness]*len(mag_m),[frequency]*len(mag_m),xm,ym,mag_m,fit().STFMR(mag_m,*para_FMRm)],\
        index = ['thickness(nm)','frequency(GHz)','negative field(mT)','V_DC_raw(uV)_n','fit_negative_field(mT)','fit_V_DC(uV)_n'])
        Raw_d_n = Q.T

        # fit parameter DataFrame
        para_FMRm = pd.DataFrame(para_FMRm,index = ['V_sym(uV)_n','ΔH(mT)_n','H_FMR(mT)_n','V_asym(uV)_n','slope(uV/mT)_n','offset(uV)_n']).T
        sigma_FMR_m = np.sqrt(np.diag(cov_FMRm))
        sigma_FMR_m = pd.DataFrame(sigma_FMR_m,index = ['V_sym(uV)_err_n','ΔH(mT)_err_n','H_FMR(mT)_err_n','V_asym(uV)_err_n',\
            'slope(uV/mT)_err_n','offset_err(uV)_n']).T
        thickness1 = pd.DataFrame([thickness],index = ['thickness(nm)']).T
        frequency1 = pd.DataFrame([frequency],index =['frequency(GHz)']).T
        fit_data_m = pd.concat([thickness1,frequency1,para_FMRm,sigma_FMR_m],axis=1)

        # parameter bounds (positive field)
        params_bounds_p =[
            [-np.inf,0,0,-np.inf,-np.inf,-np.inf],
            [np.inf,np.inf,np.inf,np.inf,np.inf,np.inf]
        ]
        # fitting for positive field
        xp=data[dN//2+25:dN,0] # raw data for positive field
        yp=data[dN//2+25:dN,1]*1e+6 # raw data for positive field
        ap = max(yp)
        a1p = np.argmax(yp)
        a2p = data[dN//2+9+a1p,0]
        bp = min(yp)
        b1p = np.argmin(yp)
        b2p = data[dN//2+9+b1p,0]
        # Symmetry, linewidth, resonance field, Antisym, slope, offset
        inip = [10, 20, (a2p+b2p)/2, np.absolute(ap-bp), 0.0001, 1]
        try:
            para_FMRp,cov_FMRp=curve_fit(fit().STFMR, xp, yp, inip, maxfev =12000, bounds =params_bounds_p)
        except RuntimeError:
            pass
        mag_p=np.arange(0,300.1,0.1)

        if Plot:
            axes[0].plot(xp,yp-para_FMRp[4]*xp-para_FMRp[5],'ok',markersize=3,label=str(thickness)+'nm'+str(frequency)+'GHz')
            axes[0].plot(mag_p,fit().STFMR(mag_p,*para_FMRp)-para_FMRp[4]*mag_p-para_FMRp[5],'-r')
            R2p = calc().R2(yp,fit().STFMR(xp,*para_FMRp))
            axes[1].plot(xp,yp-fit().STFMR(xp,*para_FMRp),'-r',label=f'R^2={R2p:.5f}')
            axes[0].set_xlim(-300,300)
            fig.legend(bbox_to_anchor=(1.0, 0.9), loc='upper left',ncol=2).get_frame().set_alpha(0)
            axes[0].set_ylabel(r'$V_{\mathrm{DC}}$ $\mathrm{(\mu V)}$')
            axes[1].set_xlabel(r'$\mu_0H$ $\mathrm{(mT)}$')
            axes[1].set_ylabel(r'$\rm{Residuals}$ $\mathrm{(\mu V)}$')


        # dataframe for positive field
        #fit data DataFrame
        Q = pd.DataFrame([xp,yp,mag_p,fit().STFMR(mag_p,*para_FMRp)],\
        index = ['positive field(mT)','V_DC_raw(uV)_p','fit_positive_field(mT)','fit_V_DC(uV)_p'])
        Raw_d_p = Q.T

        # fit parameter DataFrame
        para_FMRp = pd.DataFrame(para_FMRp,index = ['V_sym(uV)_p','ΔH(mT)_p','H_FMR(mT)_p','V_asym(uV)_p','slope(uV/mT)_p','offset(uV)_p']).T
        sigma_FMR_p = np.sqrt(np.diag(cov_FMRp))
        sigma_FMR_p = pd.DataFrame(sigma_FMR_p,index = ['V_sym(uV)_err_p','ΔH(mT)_err_p','H_FMR(mT)_err_p','V_asym(uV)_err_p',\
            'slope(uV/mT)_err_p','offset_err(uV)_p']).T
        fit_data_p = pd.concat([para_FMRp,sigma_FMR_p],axis=1)

        # conection data posi and nega field
        Raw_d = pd.concat([Raw_d_n,Raw_d_p],axis =1)
        fit_data = pd.concat([fit_data_m, fit_data_p],axis =1)


        return Raw_d,fit_data

    def STFMR_repeat_fit(self,path,txt,save=True,Plot=True):
        Raw_data = pd.DataFrame()
        Fit_para = pd.DataFrame()
        for pathname, dirnames, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith(txt):
                    path_in = os.path.join(pathname, filename)
                    p_sub=pathlib.Path(path_in) # ディレクトリの階層を上がるため
                    if str(p_sub.parents[0])[-6:]=='signal':
                        Raw_D, Fit_param=self.singlefit_process_STFMR(path_in,Plot)
                        Raw_data = pd.concat([Raw_data,Raw_D],axis=0)
                        Fit_para = pd.concat([Fit_para,Fit_param],axis=0)
                        #axes[0].xlabel(r'$\mu_0H$ $\mathrm{(mT)}$')
            plt.show()
        Fit_para = Fit_para.sort_values('thickness(nm)') # sort with thickness
        if save:
            Raw_data.to_csv(f'./result/{self.name2}/stfmr_fit_data.csv',index=False)
            Fit_para.to_csv(f'./result/{self.name1}/stfmr_fit_parameter.csv',index=False)
        return Raw_data, Fit_para


    # kittel fit process
    def kittel_fixed_process(self,data, save=True,Plot=True): # gamma(gyromagnetic ratio) fixed
        Raw_data = pd.DataFrame()
        Fit_para = pd.DataFrame()
        for thick in data['thickness(nm)'].unique():
            data_t=data[data['thickness(nm)']==thick]
            data_t=data_t.sort_values('frequency(GHz)', ascending=True)
            Hfmr_p = data_t['H_FMR(mT)_p']
            Hfmr_n = abs(data_t['H_FMR(mT)_n'])
            Hfmr_ave=(Hfmr_p+Hfmr_n)/2 # average of posi and nega field
            f = data_t['frequency(GHz)']
            ini=[800,10]
            param_bound =[[0,-np.inf],[np.inf,np.inf]]
            try:
                para_kit_ave,cov_kit_ave=curve_fit(fit().kittel_fixed,Hfmr_ave,data_t['frequency(GHz)'],ini,bounds = param_bound)
            except RuntimeError:
                pass
            H=np.arange(abs(para_kit_ave[1])+0.1,300,0.5)
            if Plot:
                plt.plot(Hfmr_ave,f,'ok',label =str(thick)+'nm')
                plt.plot(H,fit().kittel_fixed(H,*para_kit_ave),'-r')
                plt.xlabel(r'$\mu_0 H_{\rm{FMR}}$ $\rm{(mT)}$')
                plt.ylabel(r'$f ~$ $(\rm{GHz})$')
                plt.legend(loc=(1,0.3),ncol=2).get_frame().set_alpha(0)
                plt.xlim(0,300)
                plt.ylim(0,max(f)+3)
                plt.show()

            #fit data DataFrame
            Q = pd.DataFrame([[thick]*len(H),np.array(Hfmr_ave),np.array(f),H,fit().kittel_fixed(H,*para_kit_ave)],\
            index = ['thickness(nm)','H_FMR_raw(mT)','frequency_raw(GHz)','fit_H_FMR(mT)','fit_freqency(GHz)'])
            Raw_d_1 = Q.T

            # fit parameter DataFrame
            para_kittel = pd.DataFrame(para_kit_ave,index = ['Meff(mT)','Hani(mT)']).T
            sigma_kittel = np.sqrt(np.diag(cov_kit_ave))
            sigma_kittel = pd.DataFrame(sigma_kittel,index = ['Meff(mT)_err','Hani(mT)_err']).T
            thickness = pd.DataFrame([thick],index = ['thickness(nm)']).T
            fit_data = pd.concat([thickness,para_kittel,sigma_kittel],axis=1)

            Raw_data = pd.concat([Raw_data,Raw_d_1],axis = 0)
            Fit_para = pd.concat([Fit_para,fit_data],axis = 0)
        Fit_para = Fit_para.reset_index(drop =True)
        if save:
            Raw_data.to_csv(f'./result/{self.name2}/kittel_fit_data.csv',index=False)
            Fit_para.to_csv(f'./result/{self.name1}/kittel_fit_parameter.csv',index=False)
        return Raw_data, Fit_para

    # linewidth fit process

    def linewidth_process(self,data,save=True,Plot=True):
        Raw_data = pd.DataFrame()
        Fit_para=pd.DataFrame()
        for thick in data['thickness(nm)'].unique():
            data_t = data[data['thickness(nm)']==thick]
            data_t = data_t.sort_values('frequency(GHz)', ascending=True)
            linep = data_t['ΔH(mT)_p']
            linen = data_t['ΔH(mT)_n']
            line_ave=(linep+linen)/2
            x = data_t['frequency(GHz)']
            y =line_ave
            fre=np.arange(0,max(x)+4,0.1)
            ini=[1,0.01]
            para_line_ave,cov_line_ave=curve_fit(fit().linewidth, x,y,ini)
            if Plot:
                plt.plot(x,y,'ok',label=str(thick)+'nm')
                plt.plot(fre, fit().linewidth(fre,*para_line_ave),'-r')
                plt.xlabel(r'$f$ $\rm{(GHz)}$')
                plt.ylabel(r'$\Delta H~$ $(\rm{mT})$')
                plt.legend(loc=(1,0.3),ncol=2).get_frame().set_alpha(0)
                plt.xlim(0,max(x)+3)
                plt.show()
            para_line_ave[0]*=self.gamma # gyromagnetic ratio for alpha
            cov_line_ave[0][0]*=self.gamma**2 # alpha err

            #fit data DataFrame
            Q = pd.DataFrame([[thick]*len(fre),np.array(x),np.array(y),fre,fit().linewidth(fre,*para_line_ave)],\
            index = ['thickness(nm)','frequency_raw(GHz)','line_width(mT)','fit_frequency(GHz)','fit_line_width(mT)'])
            Raw_d_1 = Q.T

            # fit parameter DataFrame
            para_line_width = pd.DataFrame(para_line_ave,index = ['alpha_G','W0(mT)']).T
            sigma_line_width = np.sqrt(np.diag(cov_line_ave))
            sigma_line_width = pd.DataFrame(sigma_line_width,index = ['alpha_G_err','W0(mT)_err']).T
            thickness = pd.DataFrame([thick],index = ['thickness(nm)']).T
            fit_data = pd.concat([thickness,para_line_width,sigma_line_width],axis=1)

            Raw_data = pd.concat([Raw_data,Raw_d_1],axis = 0)
            Fit_para = pd.concat([Fit_para,fit_data],axis = 0)
        Fit_para = Fit_para.reset_index(drop =True)
        if save:
            Raw_data.to_csv(f'./result/{self.name2}/line_width_fit_data.csv',index=False)
            Fit_para.to_csv(f'./result/{self.name1}/line_width_fit_parameter.csv',index=False)
        return Raw_data, Fit_para

class torque_eff:
    def Current_est(self,Fit_para_RF, Fit_para_DC, P=100,L=135e-6):
        Cur = pd.DataFrame()
        for i in Fit_para_RF['thickness(nm)'].unique():
            b=Fit_para_RF[Fit_para_RF['thickness(nm)']==i]
            a=Fit_para_DC[Fit_para_DC['thickness(nm)']==i]
            a=a.reset_index(drop=True)
            a_DC, b_RF=a['a(ohm/(mA)^2)'][0], b['a(ohm/mW)']
            I_RF, R0 = pow(P*2*b_RF/a_DC,0.5), a['R(ohm)'][0]
            I_RF = I_RF.rename('I_RF(mA)')
            E = R0*I_RF/L *1e-3/1000 #kV/m
            E = E.rename('E (kV/m)')

            Z = pd.concat([b['thickness(nm)'], b['frequency(GHz)'],I_RF,E],axis=1)
            Cur = pd.concat([Cur,Z],axis=0)
        return Cur

    def All_data_param(self,AMR,RF,DC,STFMR,kittel, P=100, d=135e-6):
        Cur = self.Current_est(RF,DC,P,d)
        Fit_para_stfmr = STFMR.reset_index(drop=True)
        frequency =Fit_para_stfmr['frequency(GHz)'].unique()
        data1 = pd.DataFrame()
        for freq in frequency:
            Fit_tor = Fit_para_stfmr[Fit_para_stfmr['frequency(GHz)']==freq]
            Fit_tor =Fit_tor.reset_index(drop= True)
            Vs = (Fit_tor['V_sym(uV)_p']-Fit_tor['V_sym(uV)_n'])/2
            Va = (Fit_tor['V_asym(uV)_p']+Fit_tor['V_asym(uV)_n'])/2
            W = (Fit_tor['ΔH(mT)_p']+Fit_tor['ΔH(mT)_n'])/2
            HFMR = (Fit_tor['H_FMR(mT)_p']+abs(Fit_tor['H_FMR(mT)_n']))/2
            HFMR, Vs, Va, W = HFMR.rename('H_FMR(mT)'), Vs.rename('V_sym(uV)'),Va.rename('V_asym(uV)'), W.rename('W(mT)')
            d_F = Fit_tor['thickness(nm)']
            freq1= Fit_tor['frequency(GHz)']

            Meff = kittel['Meff(mT)']
            Cur_tor = Cur[Cur['frequency(GHz)']==freq]
            Cur_tor = Cur_tor.reset_index(drop=True)
            I_RF = Cur_tor['I_RF(mA)']
            E = Cur_tor['E (kV/m)']
            dR = abs(AMR['dR(ohm)'])

            data = pd.concat([d_F, dR, I_RF,E,freq1, Vs, Va, HFMR, W, Meff],axis=1)
            data1 =pd.concat([data1,data],axis=0)
        return data1

    def par_data_param(self,STFMR,kittel):
        Fit_para_stfmr = STFMR.reset_index(drop=True)
        frequency =Fit_para_stfmr['frequency(GHz)'].unique()
        data1 = pd.DataFrame()
        for freq in frequency:
            Fit_tor = Fit_para_stfmr[Fit_para_stfmr['frequency(GHz)']==freq]
            Fit_tor =Fit_tor.reset_index(drop= True)
            Vs = (Fit_tor['V_sym(uV)_p']-Fit_tor['V_sym(uV)_n'])/2
            Va = (Fit_tor['V_asym(uV)_p']+Fit_tor['V_asym(uV)_n'])/2
            W = (Fit_tor['ΔH(mT)_p']+Fit_tor['ΔH(mT)_n'])/2
            HFMR = (Fit_tor['H_FMR(mT)_p']+abs(Fit_tor['H_FMR(mT)_n']))/2
            HFMR, Vs, Va, W = HFMR.rename('H_FMR(mT)'), Vs.rename('V_sym(uV)'),Va.rename('V_asym(uV)'), W.rename('W(mT)')
            d_F = Fit_tor['thickness(nm)']
            freq1= Fit_tor['frequency(GHz)']

            Meff = kittel['Meff(mT)']

            data = pd.concat([d_F,freq1, Vs, Va, HFMR, W, Meff],axis=1)
            data1 =pd.concat([data1,data],axis=0)
        return data1

    def xi_fmr_calc(self,data,d_N,Ms):
        Vs, Va, HFMR, Meff,d_F =data['V_sym(uV)'],data['V_asym(uV)'], \
            data['H_FMR(mT)'], data['Meff(mT)'],data['thickness(nm)']
        xi_FMR = fit().xi_FMR2(Va,Vs,Ms/1000,d_F*1e-9,d_N*1e-9,Meff,HFMR)
        xi_FMR = xi_FMR.rename('ξ_FMR')
        result = pd.concat([data['frequency(GHz)'],data['thickness(nm)'],xi_FMR],axis =1)
        return result
    
    def xi_fmr_single(self,data,d_N,Ms,d_F):
        Vs, Va, HFMR, Meff =data['V_sym(uV)'],data['V_asym(uV)'], \
            data['H_FMR(mT)'], data['Meff(mT)']
        xi_FMR = fit().xi_FMR2(Va,Vs,Ms/1000,d_F*1e-9,d_N*1e-9,Meff,HFMR)
        xi_FMR = xi_FMR.rename('ξ_FMR')
        result = pd.concat([data['frequency(GHz)'],xi_FMR],axis =1)
        return result

    def H_DL_calc(self,data):
        Vs, HFMR, W, Meff,dR,I_RF =data['V_sym(uV)'], data['H_FMR(mT)'], data['W(mT)'], data['Meff(mT)'],\
            data['dR(ohm)'],data['I_RF(mA)']
        H_DL = abs(fit().H_DL(Vs, HFMR, W, Meff, dR, I_RF))
        H_DL = H_DL.rename('H_DL(mT)')
        result = pd.concat([data['frequency(GHz)'],data['thickness(nm)'],H_DL],axis =1)
        return result

    def H_parallel_calc(self,data):
        Va, HFMR, W, Meff,dR,I_RF =data['V_asym(uV)'], data['H_FMR(mT)'], data['W(mT)'], data['Meff(mT)'],\
            data['dR(ohm)'],data['I_RF(mA)']
        H_parallel = abs(fit().H_parallel(Va, HFMR, W, Meff, dR, I_RF))
        H_parallel = H_parallel.rename('H_parallel(mT)')
        result = pd.concat([data['frequency(GHz)'],data['thickness(nm)'],H_parallel],axis =1)
        return result


class stfmr_dep_ana:
    def __init__(self,):
        pass

    # stfmr angular dependence


    # stfmr power dependence

if __name__ == '__main__':
    # ディレクトリの指定
    # 現在のフォルダからデータフォルダへの相対パス
    Path_data=os.path.relpath('../data/Ti(1)Pt(8)Ni(x)SiO2(4)/', './')
    target_dir =Path_data+'/'
    STFMR = stfmr_parameter()
    Raw_data,Fit_para = STFMR.AMR_repeat_fit(target_dir,'.txt')
    Fit_para.to_csv('AMR_parameter.csv',index=False)