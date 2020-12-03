
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigs
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import csv
import xlwt
def readCsv(csvfname):
    # read csv to list of lists
    with open(csvfname, 'r') as csvf:
        reader = csv.reader(csvf)
        csvlines = list(reader)
    return csvlines


    
class SurvivalMetric(object):
    def __init__(self, severity_cls_preds, risk_reg_preds, gt_severity, gt_hitday, gt_event_indicator, thres=0.5):
        self.severity_cls_preds = severity_cls_preds  
        self.gt_severity = np.where(gt_severity<=1, 0, 1)    #true  label matric
        self.risk_reg_preds = risk_reg_preds
        self.gt_hitday = gt_hitday
        self.pred_hitday = np.argmax(self.risk_reg_preds,axis = -1)
        self.event_indicator = gt_event_indicator
        
        self.CIF = self.get_CIF()
        self.update_confusion_matrix(thres)
        
    def TDC_index(self):
        N = self.risk_reg_preds.shape[0]
        A = 0
        F = 0
        for i in range(N):
            if self.event_indicator[i]==0 or self.event_indicator[i]==2:
                continue
            for j in range(i+1, N):
                if self.gt_hitday[i]<self.gt_hitday[j]:
                    A+=1
                    if self.CIF[i, self.gt_hitday[i]]>self.CIF[j,self.gt_hitday[i]]:
                        F+=1
        return F/(A+0.000001)
    
    def MeanHitDiff(self):
        N = self.risk_reg_preds.shape[0]
        diff_sum = 0
        for i in range(N):
            if self.event_indicator[i]==0 or self.event_indicator[i]==2:
                continue
            distribution = self.risk_reg_preds[i,...]
            pred_hitday = np.argmax(distribution)+1
            gt_hitday = self.gt_hitday[i]
            diff_sum+= abs(pred_hitday-gt_hitday)
        return diff_sum/N
        
    def get_CIF(self):
        H, W = self.risk_reg_preds.shape
        triu_mask = np.triu(np.ones((W,W), dtype=np.float32),0)                 
        CIF =np.matmul(self.risk_reg_preds, triu_mask)
        return CIF
   
        
    def accuracy(self):        
        return (self.tps+self.tns)/(self.tns+self.tps+self.fns+self.fps)

    def sensitivity(self):
        return self.tps/(self.tps+self.fns)
    
    def sensitivity_dead(self):
        pred_severity = self.pred_severity.copy()
        pred_severity[self.event_indicator!=2] = 0
        return np.sum(pred_severity==1)/np.sum(self.event_indicator==2)

    def precision(self):
        return self.tps/(self.tps+self.fps)
    
    def specificity(self):
        return self.tns/(self.tns+self.fps)
    
    def f1value(self):
        return 2*(self.precision()*self.sensitivity())/(self.precision()+self.sensitivity())
    
    def get_optimal_thres(self):
        optimal_thres = 0.01
        optimal_f1value = 0
        for thres in np.arange(0.01, 1.0, 1.0/100):
            self.update_confusion_matrix(thres)
            f1value = self.f1value()
            if f1value>optimal_f1value:
                optimal_f1value=f1value
                optimal_thres = thres
        return optimal_thres
            
    def update_confusion_matrix(self, thres):
        self.pred_severity = np.where(self.severity_cls_preds[:,-1]>thres, 1, 0)       
        
        self.tps = np.sum(self.pred_severity*self.gt_severity)
        self.fps = np.sum(self.pred_severity*(1-self.gt_severity))
        self.tns = np.sum((1-self.pred_severity)*(1-self.gt_severity))
        self.fns = np.sum((1-self.pred_severity)*self.gt_severity)
    
    def auROC(self):         
        fpr, tpr, _ = roc_curve(self.gt_severity, self.severity_cls_preds[:,-1]) 
        roc_auc  = auc(fpr, tpr)
        return fpr, tpr, roc_auc
       
import glob
import os.path as osp
import matplotlib.pyplot as plt

def compute_survival_metrics(with_treatment=True):
    if with_treatment:
        nettype = 'eval_results/WithTreatment'
    else:
        nettype = 'eval_results/WithoutTreatment'
        
    subsets = ['subset_1', 'subset_2', 'subset_3', 'subset_4', 'subset_5']
    cls_type = 'C2'
    mACCs = []
    mSENs = []
    mDEADSENs = []
    mPREs = []    
    mSPEs = []
    mAUCs = []
    mCIs = []
    fprs = []
    tprs = []
    mHDIFFs = []
    curves = []
    plt.figure()
    font = {'family' : 'serif',  
        'weight' : 'light',  
        'size'   : 15,  
        } 
    
    colors = ['red', 'green', 'blue','black','aqua','darkviolet']
    cohorts = ['Cohort_1','Cohort_2','Cohort_3','Cohort_4','Cohort_5']
    for i, subset in enumerate(subsets):
        print('============={0}==========='.format(subset)) 
        severity_preds = np.array(readCsv(glob.glob(osp.join(nettype, '*'+subset+'*severity_cls_preds.csv'))[0])[1:], dtype=np.float32)
        gt_severity = np.array(readCsv(glob.glob(osp.join(nettype, '*'+subset+'*gt_severity.csv'))[0])[1:], dtype=np.int32).reshape(-1)
        risk_preds = np.array(readCsv(glob.glob(osp.join(nettype, '*'+subset+'*risk_reg_preds.csv'))[0])[1:], dtype=np.float32)
        gt_hitday = np.array(readCsv(glob.glob(osp.join(nettype, '*'+subset+'*gt_hitday.csv'))[0])[1:], dtype=np.int32).reshape(-1)
        event_indicators = np.array(readCsv(glob.glob(osp.join(nettype, '*'+subset+'*indicator.csv'))[0])[1:], dtype=np.int32).reshape(-1)
        
        myMetic = SurvivalMetric(severity_preds, risk_preds, gt_severity, gt_hitday, event_indicators, 0.5)
        thres = myMetic.get_optimal_thres()
        print('optimal thres:', thres)
        myMetic.update_confusion_matrix(thres)
        mACCs.append(myMetic.accuracy())
        mSENs.append(myMetic.sensitivity())
        mDEADSENs.append(myMetic.sensitivity_dead())
        mPREs.append(myMetic.precision())
        mSPEs.append(myMetic.specificity())
        mCIs.append(myMetic.TDC_index())
        mHDIFFs.append(myMetic.MeanHitDiff())
        fpr, tpr, auc = myMetic.auROC()
        mAUCs.append(auc)
        fprs.append(fpr)
        tprs.append(tpr)
        A,=plt.plot(fpr, tpr, color=colors[i], linestyle='-', lw=2, label= cohorts[i]+'(AUC = %0.3f)' % auc)
        curves.append(A)
      
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.yticks([i*0.05 for i in range(0,21)])
    plt.xlabel('False Positive Rate',fontsize=13)
    plt.ylabel('True Positive Rate',fontsize=13)
    plt.title('ROCs of the high-risk patients classification',fontsize=15)
    plt.legend(loc="lower right")
    plt.legend(handles=curves,prop=font)
    plt.savefig(nettype+'_roc.png', dpi=300)
   
    mACCs = np.array(mACCs)
    mSENs = np.array(mSENs)
    mDEADSENs = np.array(mDEADSENs)
    mPREs = np.array(mPREs)
    mSPEs = np.array(mSPEs)
    mCIs = np.array(mCIs) 
    mHDIFFs = np.array(mHDIFFs) 
    mAUCs = np.array(mAUCs)   
   
    macc_foldmean = np.mean(mACCs, axis=0)
    msen_foldmean = np.mean(mSENs, axis=0)
    mdeadsen_foldmean = np.mean(mDEADSENs, axis=0)
    mpre_foldmean = np.mean(mPREs, axis=0)
    mspe_foldmean = np.mean(mSPEs, axis=0)
    mci_foldmean = np.mean(mCIs, axis=0)
    mhdiff_foldmean = np.mean(mHDIFFs, axis=0)
    mauc_foldmean = np.mean(mAUCs, axis=0)   

       
    macc_foldstd = np.sqrt(np.var(mACCs, axis=0))
    msen_foldstd = np.sqrt(np.var(mSENs, axis=0))
    mdeadsen_foldstd = np.sqrt(np.var(mDEADSENs, axis=0))
    mpre_foldstd = np.sqrt(np.var(mPREs, axis=0))
    mspe_foldstd = np.sqrt(np.var(mSPEs, axis=0))
    mci_foldstd = np.sqrt(np.var(mCIs, axis=0))
    mhdiff_foldstd = np.sqrt(np.var(mHDIFFs, axis=0))
    mauc_foldstd = np.sqrt(np.var(mAUCs, axis=0))
    
   
    mean_metrics = np.array([macc_foldmean, msen_foldmean, mdeadsen_foldmean, mpre_foldmean, mspe_foldmean, mauc_foldmean, mci_foldmean, mhdiff_foldmean])
    std_metrics = np.array([macc_foldstd, msen_foldstd, mdeadsen_foldstd, mpre_foldstd, mspe_foldstd, mauc_foldstd, mci_foldstd, mhdiff_foldstd])
    print('mean_metrics:', mean_metrics)
    
    cls_metrics = np.concatenate([np.expand_dims(mACCs, axis=-1), \
                                  np.expand_dims(mSENs, axis=-1), \
                                  np.expand_dims(mDEADSENs, axis=-1), \
                                  np.expand_dims(mPREs, axis=-1), \
                                  np.expand_dims(mSPEs, axis=-1), \
                                  np.expand_dims(mAUCs, axis=-1), \
                                  np.expand_dims(mCIs, axis=-1) , \
                                  np.expand_dims(mHDIFFs, axis=-1)], axis=-1)
    print(cls_metrics.shape)
    #saving to csv
    f = xlwt.Workbook()
    sheet1 = f.add_sheet(u'sheet1',cell_overwrite_ok=True)
    
    metric_headers = ['accuracy', 'sensitivity', 'dead sensitivity', 'precision', 'specificity' ,'auc', 'CI', 'MeanDiff']
    subsets = ['subset 1', 'subset 2', 'subset 3', 'subset 4', 'subset 5', 'overall']
    for i, metric_name in enumerate(metric_headers):      
        sheet1.write(0, i+1, metric_name)            
        scale = 100
        if i == 6 or i ==7:
            scale =1
        for j, subset in enumerate(subsets):
            sheet1.write(j+1, 0, subset)       
            if j==5:
                metric_v = "{:.3f}".format(mean_metrics[i]*scale) + " ± " + "{:.3f}".format(std_metrics[i]*scale) 
                sheet1.write(j+1, i+1, metric_v)
            else:
                metric_v = "{:.3f}".format(cls_metrics[j][i]*scale)
                sheet1.write(j+1, i+1, metric_v)
                
     
    save_name = osp.basename(nettype)+'_'+cls_type+'_survival_metrices.xls'
    save_name = save_name.replace('*', '')
    f.save(save_name)
    
if __name__ == '__main__':
    compute_survival_metrics(True)
    compute_survival_metrics(False)