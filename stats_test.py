import numpy as np
import pandas as pd
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sa
import statsmodels.formula.api as sfa
import scipy.stats as ss
import statsmodels.stats.multicomp as sm

def normal_dist(data,val_col,group_col,result=False):
    """
    data : pd.DataFrame

    val_col : str
            The name of columns which you test.

    group_col : strings or list
            The name of columns which you want to devide data.


    result = True -> Show p-value and result

    """

    keys = data.groupby(group_col).groups.keys()
    pw = 1
    for key in keys:
        #
        data_list = [data.loc[data.groupby(group_col).groups[key], val_col].values]
        w,pw = ss.shapiro(data_list)
        if pw < 0.05:
            print("Non-parametric")
            if result == True:
                print("""
=====================================================
p-value from [{}]is {} . Null hypothesis is rejected.
These data wouldn't be normal distribution.
You should use Non parametric / from shapiro test.
=====================================================""".format(key,pw))
            break
    if pw >0.05:
            print("Parametric")
            if result == True:
                print("""
===============================================
p-value is {}. Null hypothesis is not rejected.
These data wouldn be normal distribution.
Pls use parametric / from shapiro test.
===============================================""".format(pw))
    return pw




def homo_variance(data,val_col,group_col,result=False):
    """
    data : pd.DataFrame

    val_col : str
            The name of columns which you test.

    group_col : strings or list
            The name of columns which you want to devide data.

    **kwargs : bool

    result = True -> Show p-value and result

    """

    one_d_data = [data.loc[ids,val_col].values for ids in data.groupby(group_col).groups.values()]
    pw_normal_dist = normal_dist(data,val_col,group_col,result=result)
    if pw_normal_dist > 0.05:
        stastic,p_value = ss.bartlett(*one_d_data)
        if p_value > 0.05:
            print("Equal variance")
            if result == True:
                print("""
===========================================================
p-value is {}. Null hypothesis is not rejected.
These data's variance are not different. / from levene test
===========================================================""".format(p_value))
        if p_value < 0.05:
            print("Unequal variance")
            if result == True:
                print("""
===========================================================
p-value is {}. Null hypothesis is not rejected.
These data's variance are not the same. / from levene test
===========================================================""".format(p_value))
        return pw_normal_dist,p_value
    if pw_normal_dist < 0.05:
        stastic,p_value = ss.levene(*one_d_data)
        if p_value > 0.05:
            print("Equal variance")
            if result == True:
                print("""
===========================================================
p-value is {}. Null hypothesis is not rejected.
These data's variance are not different. / from fligner test
===========================================================""".format(p_value))
        if p_value < 0.05:
            print("Unequal variance")
            if result == True:
                print("""
===========================================================
p-value is {}. Null hypothesis is not rejected.
These data's variance are not the same. / from fligner test
===========================================================""".format(p_value))
        return pw_normal_dist,p_value



def one_way_ANOVA(data,val_col,group_col,result=False):
    """
    正規性、等分散性からANOVAもしくはKuraskal-Wallisを行う。
    return : 分散分析のp値、使うべき検定名
    検定名は任意に変更してください。 (ex. "HSD" → "scheffe")
    sign_barplot()で検定名を参照します。
    """


    ##正規性と等分散性が棄却される確立
    pw_normal,pw_varian = homo_variance(data,val_col,group_col,result=result)
    if pw_normal > 0.05:
        if pw_varian > 0.05:
            #正規分布で等分散 -> ANOVA
            lm = sfa.ols('{} ~ {}'.format(val_col,group_col), data=data).fit()
            anova = sa.stats.anova_lm(lm)                       #anovaの検定
            p = anova["PR(>F)"][group_col]                      #p値

            ##結果の表示
            print("""
Result from ANOVA
=======================================================
{}""".format(anova))

            if p <0.05:                                         #有意差がある場合
                print("""
These data posess significant difference!!!!!!!!
Recommend to use [Tukey HSD test], [Tukey test] or [scheffe test]""")
                statical_test = "HSD"                           #Tukey_HSD検定を返す（任意で変更可）
                return p,statical_test


            if p >0.05:                                         #有意差が無い場合
                print("""
No significant difference among these data. m(_ _)m
But you can try some non-parametric test or Games–Howell test.
ex.) [Steel-Dwass(dscf) test] or [conover test]""")
                statical_test = "dscf"                          #Steel-Dwass検定を返す（任意で変更可）
                return p,statical_test



    if pw_normal <0.05 or pw_varian<0.05:
        #分散が等しくない or 非正規分布
        data = [data.loc[ids,val_col].values for ids in data.groupby(group_col).groups.values()]
        H, p = ss.kruskal(*data)                               #Kuraskal-Wallis検定を実行
        print("p =",p,",H =",H," / from Kruskal-Wallis test")
        if p <0.05:                                            #有意差がある場合
            print("""
===========================================================
p-value is {}.
These data posess significant difference!!!!!!!
Recomend to use [Steel-Dwass(dscf) test] or [conover test].""".format(p))
            statical_test = "conover"                          #conover検定を返す（任意で変更可）
            return p,statical_test


        if p >0.05:                                           #有意差が無い場合
            print("""
===========================================================
p-value is {}.
No significant difference among these data. m(_ _)m
Buy you can try some non-parametric test.
ex.) [Steel-Dwass(dscf) test] or [conover test]""".format(p))
            statical_test = "dscf"                             #Steel-Dwass検定を返す（任意で変更可）
            return p,statical_test



def tukey_hsd(df,val_col,group_col):
    #dataを検定に使えるように整形
    keys = df.groupby(group_col).groups.keys()
    val_data =[]
    group_data =[]
    for key in keys:
        d = df.loc[df.groupby(group_col).groups[key],val_col]
        val_data.append(d)
        group_data.append([key]*len(d))
    result = sm.pairwise_tukeyhsd(np.concatenate(val_data), np.concatenate(group_data))

    #結果をDataFrameに変換
    groups = np.array(result.groupsunique, dtype=np.str)
    groups_len = len(groups)
    vs = pd.DataFrame(np.zeros(groups_len*groups_len).reshape(groups_len,groups_len))
    for a in result.summary()[1:]:
        a0 = str(a[0])
        a1 = str(a[1])
        a0i = np.where(groups == a0)[0][0]
        a1i = np.where(groups == a1)[0][0]
        vs[a0i].loc[a1i] = a[3].data
        vs[a1i].loc[a0i] = a[3].data
    vs.index,vs.columns = groups,groups
    return vs



def sign_barplot(df,val_col,group_col,test="HSD"):
    if test == "HSD":
        result_df = tukey_hsd(df,val_col,group_col)
    if test == "tukey":
        result_df = sp.posthoc_tukey(df,val_col,group_col)
    if test == "ttest":
        result_df = sp.posthoc_ttest(df,val_col,group_col)
    if test == "scheffe":
        result_df = sp.posthoc_scheffe(df,val_col,group_col)
    if test == "dscf":
        result_df = sp.posthoc_dscf(df,val_col,group_col)
    if test == "conover":
        result_df = sp.posthoc_conover(df,val_col,group_col)
    #マッピングのプロファイル
    fig,ax = plt.subplots(1,2,figsize=(10,6))
    cmap = ['1', '#fb6a4a',  '#08306b',  '#4292c6', '#c6dbef']
    heatmap_args = {'cmap':cmap,'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True}

    sp.sign_plot(result_df,ax=ax[1], **heatmap_args)                   #検定結果を描画

    sns.barplot(data=df,x=group_col,y=val_col,capsize=0.1,ax=ax[0])    #使ったデータを描画
    plt.show()


def stats_test(df,val_col,group_col,test=False,result=False):
    #テスト名を指定できる。指定したらすべてのtestが指定した検定に変わる
    if not test:
        p,test = one_way_ANOVA(df,val_col,group_col,result=result)
    print("This result calculated by {} test.".format(test))
    sign_barplot(df,val_col,group_col,test=test)
