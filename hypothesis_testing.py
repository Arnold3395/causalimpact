import numpy as np
import pandas as pd
from scipy.stats import normaltest,shapiro,kstest,skewtest,anderson,ttest_1samp,ttest_ind,ttest_rel
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

def normal_test(data):
    '''
    This function performs normality test on each column of the data.
    The test includes:
    1. Normal Test
    2. Shapiro Test
    3. Kolmogorov-Smirnov Test
    4. Skew Test
    '''
    result = {}
    if type(data) == pd.core.frame.DataFrame:
        colnames = list(data.columns)
    elif type(data) == np.ndarray:
        data = pd.DataFrame(data)
        colnames = list(data.columns)
    elif type(data) == list:
        data = pd.DataFrame(data)
        colnames = list(data.columns)
    else:
        print("Error: data type not supported!")
        return None
    for col in colnames:
        # initialize the result
        result[col] = {}
        x = data.loc[:,col]
        # calculate the statistic and p-value
        nt_stat, nt_p = normaltest(x)
        shapiro_stat,shapiro_p = shapiro(x)
        ks_stat,ks_p = kstest((x - np.mean(x))/np.std(x),'norm')
        skew_stat,skew_p = skewtest(x)
        # print the result
        print("============= normal test ===========")
        print("============= Column: %s  ==========="%col)
        print("Normal  Test, statistic_value: %.4f, p_value: %.4f"%(nt_stat,nt_p))
        print("Shapiro Test, statistic_value: %.4f, p_value: %.4f"%(shapiro_stat,shapiro_p))
        print("KS      Test, statistic_value: %.4f, p_value: %.4f"%(ks_stat,ks_p))
        print("Skew    Test, statistic_value: %.4f, p_vlaue: %.4f\n"%(skew_stat,skew_p))
        # save the result
        result[col]['normaltest'] = {'stat':nt_stat,'p':nt_p}
        result[col]['shapiro'] = {'stat':shapiro_stat,'p':shapiro_p}
        result[col]['kstest'] = {'stat':ks_stat,'p':ks_p}
        result[col]['skewtest'] = {'stat':skew_stat,'p':skew_p}
    return result
    
        
def t_test_1sample(x,mu_0,alternative = 'two-sided'):
    '''
    This function performs one sample t test on the data.
    The test includes:
    1. t test
    2. cohen's d
    '''
    # parameters check
    if alternative not in ['two-sided','less','greater']:
        print("Error: alternative should be one of 'two-sided','less','greater'!")
        return None
    # assumption check
    print("============= Assumptions Check ============")
    print("sample size: %s"%len(x))
    normal_test(x)
    # calculate the statistic and p-value
    t_statistic, p_value = ttest_1samp(x,popmean=mu_0,alternative=alternative)
    cohen_d = (np.mean(x)-mu_0)/np.std(x)
    print("==============  t Test Result  =============")
    print("t   value: %.4f"%t_statistic)
    print("p   value: %.4f"%p_value)
    print("cohen's d: %.4f\n"%cohen_d)
    return (t_statistic,p_value,cohen_d)

def t_test_ind(x1,x2,alternative = 'two-sided'):
    '''
    This function performs independent sample t test on the data.
    The test includes:
    1. t test
    2. cohen's d
    '''
    # parameters check 
    if alternative not in ['two-sided','less','greater']:
        print("Error: alternative should be one of 'two-sided','less','greater'!")
        return None
    # assumption check
    print("============= Assumptions Check ============")
    print("sample size, x1:%s, x2:%s"%(len(x1),len(x2)))
    normal_test(x1)
    normal_test(x2)
    # calculate the statistic and p-value
    t_statistic, p_value = ttest_ind(x1,x2,alternative=alternative)
    cohen_d = (np.mean(x1)-np.mean(x2))/np.sqrt((np.std(x1)**2*(len(x1))+np.std(x2)**2*len(x2))/(len(x1)+len(x2)-2))
    print("==============  t Test Result  =============")
    print("t   value: %.4f"%t_statistic)
    print("p   value: %.4f"%p_value)
    print("cohen's d: %.4f\n"%cohen_d)
    return (t_statistic,p_value,cohen_d)


def t_test_matched_sample(x1,x2,alternative = 'two-sided'):
    '''
    This function performs matched sample t test on the data.
    The test includes:
    1. t test
    2. cohen's d
    '''
    # parameters check
    if alternative not in ['two-sided','less','greater']:
        print("Error: alternative should be one of 'two-sided','less','greater'!")
        return None
    # assumption check
    print("============= Assumptions Check ============")
    print("sample size: %s"%len(x1))
    normal_test(x1-x2)
    # calculate the statistic and p-value
    t_statistic, p_value = ttest_rel(x1,x2,alternative=alternative)
    cohen_d = np.mean(x1-x2)/np.std(x1-x2)
    print("==============  t Test Result  =============")
    print("t   value: %.4f"%t_statistic)
    print("p   value: %.4f"%p_value)
    print("cohen's d: %.4f\n"%cohen_d)
    return (t_statistic,p_value,cohen_d)


