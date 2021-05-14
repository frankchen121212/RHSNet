import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
from lib.paths import *
from pandas import DataFrame
import numpy as np
from tqdm import tqdm
import seaborn as sns
import json as js
import os,sys
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,root)
# yellow -> green -> blue  -> red  -> purple
# CNN(no aug)-> CNN       -> Equi_CN     -> Seq_Model-> Seq_Chip
# palette = sns.color_palette("cubehelix_r",n_colors=6)
# color = ["#EF7D24","#2DA34F","#2177B3","#D42727","purple"]
# palette = sns.color_palette(color)
palette = sns.hls_palette(5,l=.5,s=.9)
marker_list = ['x','*','p','d','o','s']
linestyle_list = [':',':','-.','-.','-','-']
# -      实线(solid)
# --     短线(dashed)
# -.     短点相间线(dashdot)
# ：    虚点线(dotted)

def get_chromosome_distribution_img(args, summary_path):
    print("=>getting chromosome distribution of data")
    raw_datas = js.load(open(summary_path, 'r'))
    raw_datas = sorted(raw_datas, key=lambda d: d["chromosome"])
    # dic keys
    # ["chromosome"]["label"]["seqence"]["rate"]["length"]
    # ["start_index"]["end_index"]
    chromosome = {}
    for i in range(1,23):
        chromosome["chr{}".format(i)] = []
    with tqdm(total=len(raw_datas)) as t :
        for d in raw_datas:
            chr = d["chromosome"]
            chromosome[chr].append(d)
            t.update()
    chromosome_img_root = get_chromosome_img_root(args)
    #TODO the example chromosome name is defined
    example_chromosome = 12

    for i in range(example_chromosome,example_chromosome+2):
        print("=>drawing distributions for Chromosome {}".format(i+1))
        chr_list_raw = chromosome["chr{}".format(i)]
        chr_list_raw = sorted(chr_list_raw, key=lambda d: d["end_index"])

        chr_list = []
        end = chr_list_raw[0]["end_index"]
        for c in chr_list_raw:
            start = c["start_index"]
            if start != end:
                chr_list.append({"rate":0,
                                 "start_index":end,
                                 "end_index":start}
                                )
                end = c["end_index"]
                continue
            # When rate>100,it makes the bar too high and not easy to draw
            # Setting to 0
            elif c["rate"]>100:
                chr_list.append({"rate": 0,
                                 "start_index": end,
                                 "end_index": start}
                                )
            else:
                chr_list.append(c)

        fig = plt.figure(figsize=(40, 20))
        bar_plt = fig.add_subplot(111)
        rate_bar = (np.array([chr["rate"] for chr in chr_list])).astype(np.int32)
        X = ["%e" % (int(c["end_index"] - c["start_index"]/10000000)*10000000) for c in chr_list]
        x = range(len(X))
        bar_plt.bar(x, rate_bar, width=10,linewidth=5, color="black")

        plt.xticks(x, X)
        bar_plt.xaxis.set_major_locator(ticker.MultipleLocator((len(x)/5)))

        plt.ylim(0, max(rate_bar))
        plt.ylabel("Recombination rate (cM ${Mb^-1}$)")
        plt.title("Chromosome {}".format(i+1))
        fig_name = os.path.join(chromosome_img_root,"Chromosome_{}".format(i+1))
        plt.rc('font', family='DejaVu Sans')
        plt.savefig(fig_name)
        # plt.cla()
        print("=> {} saved".format(fig_name))

def Save_model_Length_Comparision(arg_list,interval,image_name,eval_key):
    plt.cla()
    fontsize = 60
    sns.set_style("white")
    font = {'family': 'DejaVu Sans', 'size': fontsize}
    matplotlib.rc('font', **font)  # set all the word into font size
    arg_example = arg_list[0]
    source_file_root = os.path.join(root, 'output', arg_example["dataset"])
    if not os.path.exists(source_file_root):
        raise Exception("You Do Not have any output yet !")
    else:
        fig = plt.figure(figsize=(65, 40))
        line_plt = fig.add_subplot(111)
        for tick in line_plt.get_xticklabels():
            tick.set_rotation(45)
        bar_plt = line_plt.twinx()
        line_plt.tick_params(axis='y',labelsize= fontsize*1.5)
        line_plt.set_ylabel(eval_key,fontsize= fontsize)
        bar_plt.set_ylabel("Num_data",fontsize= fontsize)
        X,Y =[],[]
        Num_hot,Num_cold=[],[]
        for i_arg, args in enumerate(arg_list):
            model_name = args["model"]
            perfix_name = args["output_prefix"]
            result_file_name = os.path.join(source_file_root,
                                            model_name,
                                            str(args["input_length"]) + "_" + str(args["input_height"]),
                                            perfix_name + '_result.json')
            if not os.path.exists(result_file_name):
                continue
            eval_list = js.load(open(result_file_name, 'r'))
            num_group = len(eval_list)

            len_list = [args["min_length"] + i * interval
                        for i in range(num_group)]

            X = ['[' + str(l) + ',' + str(l + interval) + ']' for l in len_list]
            Y = [eval_list[i][eval_key] for i in range(num_group)]
            Num_hot = [(eval_list[i]["TP"]+eval_list[i]["FN"])
                   for i in range(num_group)]
            Num_cold = [(eval_list[i]["TN"] +eval_list[i]["FP"])
                       for i in range(num_group)]
            line_plt.plot(X, Y ,
                          label=str(args["output_prefix"].replace('_',' ')),
                          ls = linestyle_list[i_arg],
                          lw=8,
                          )

            line_plt.scatter(X, Y,
                             marker=marker_list[i_arg],
                             s=500
                             )
            # Draw: Y*2 instead of Y to set the line higer than bar
            # Show the number of each linet
            # for a, b in zip(X, Y):
            #     line_plt.text(a, b, '%.2f' % b, ha='center', va='bottom',fontsize = 0.85*fontsize)
        line_plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3,fontsize =  fontsize*1.5)
        x = range(len(X))
        bar_plt.set_ylim(0,int(2*max(Num_hot)))
        bar_plt.bar(x, Num_hot,width=0.2, alpha=0.5,label = "Num hot",color="r")
        bar_plt.bar([i+0.2 for i in x], Num_cold, width=0.2, alpha=0.5,label = "Num cold",color="b")
        plt.xticks(x, X)
        # for a, hot_num,cold_num in zip(x, Num_hot,Num_cold):
        #     bar_plt.text(a, 0, hot_num, ha='center', va='bottom',color = "r",fontsize = 0.85*fontsize)
        #     bar_plt.text(a, 150, cold_num, ha='center', va='bottom',color = "b",fontsize = 0.85*fontsize)
        plt.xticks(fontsize= fontsize)
        plt.yticks(fontsize= fontsize)
        plt.legend(loc="upper left",fontsize =  fontsize)
        if not os.path.exists(os.path.dirname(image_name)):
            os.makedirs(os.path.dirname(image_name))
        print("=> saving {}".format(image_name))
        plt.savefig(image_name,bbox_inches='tight')

def Save_model_Violin_Comparision(final_dic, final_result_root):
    data = DataFrame(final_dic)
    Sn_data     = data[data["Eval"] == "Sn"]
    Sp_data     = data[data["Eval"] == "Sp"]
    Acc_data    = data[data["Eval"] == "Acc"]
    Mcc_data    = data[data["Eval"] == "Mcc"]
    Rec_data    = data[data["Eval"] == "Recall"]
    Prec_data   = data[data["Eval"] == "Precision"]
    F1_data     = data[data["Eval"] == "F1"]

    plt.figure(figsize=(15, 15))
    plt.tick_params(axis='y',labelsize=50)
    plt.tick_params(axis='x', labelsize=30)
    # sns.set(font_scale=4.2)
    sns.set_style("white")

    #Sn
    Boxplot(Sn_data, final_result_root, eval="Sn")
    # Sp
    Boxplot(Sp_data, final_result_root, eval="Sp")
    # Acc
    Boxplot(Acc_data, final_result_root, eval="Acc")
    # Mcc
    Boxplot(Mcc_data, final_result_root, eval="Mcc")
    # Recall
    Boxplot(Rec_data, final_result_root, eval="Recall")
    # Precision
    Boxplot(Prec_data, final_result_root, eval="Precision")
    # F1
    Boxplot(F1_data, final_result_root, eval="F1 Score")

def Boxplot(data, final_result_root, eval=None):
    ymajorLocator = MultipleLocator(0.05)  # 将y轴主刻度标签设置为0.5的倍数
    ymajorFormatter = FormatStrFormatter('%1.2f')  # 设置y轴标签文本的格式
    plt.cla()
    sns.stripplot()
    box_plt = sns.boxplot(x="Model", y="Value",
                          data=data,
                          width=0.4)
    box_plt.yaxis.set_major_locator(ymajorLocator)
    box_plt.yaxis.set_major_formatter(ymajorFormatter)

    font = {'family': 'DejaVu Sans'}
    matplotlib.rc('font', **font)
    box_fig = box_plt.get_figure()
    box_fig_file_name = os.path.join(final_result_root, eval + '.png')
    print("=> saving {}".format(box_fig_file_name))
    box_fig.savefig(box_fig_file_name,bbox_inches='tight')


def Save_model_Line_Comparision(arg_list ,interval, final_result_root):
    Sp_comparison_file_name = os.path.join(
        final_result_root,'Sp.png'
    )
    Sn_comparison_file_name = os.path.join(
        final_result_root, 'Sn.png'
    )
    Acc_comparison_file_name = os.path.join(
        final_result_root ,'Acc.png'
    )
    Mcc_comparison_file_name = os.path.join(
        final_result_root ,'Mcc.png'
    )

    Recall_comparison_file_name = os.path.join(
        final_result_root ,'Recall.png'
    )
    Precision_comparison_file_name = os.path.join(
        final_result_root ,'Precision.png'
    )
    F1_comparison_file_name = os.path.join(
        final_result_root ,'F1.png'
    )
    Save_model_Length_Comparision(arg_list=arg_list, interval=interval, image_name=Sp_comparison_file_name,
                                  eval_key="Sp")
    Save_model_Length_Comparision(arg_list=arg_list,interval=interval, image_name=Sp_comparison_file_name, eval_key="Sp")
    Save_model_Length_Comparision(arg_list=arg_list,interval=interval,image_name=Sn_comparison_file_name, eval_key="Sn")
    Save_model_Length_Comparision(arg_list=arg_list,interval=interval, image_name=Acc_comparison_file_name, eval_key="Acc")
    Save_model_Length_Comparision(arg_list=arg_list,interval=interval, image_name=Mcc_comparison_file_name, eval_key="Mcc")

    Save_model_Length_Comparision(arg_list=arg_list,interval=interval, image_name=Recall_comparison_file_name, eval_key="Recall")
    Save_model_Length_Comparision(arg_list=arg_list,interval=interval, image_name=Precision_comparison_file_name, eval_key="Precision")
    Save_model_Length_Comparision(arg_list=arg_list,interval=interval, image_name=F1_comparison_file_name, eval_key="F1")

def Save_model_ROC_Comparision(roc_list,model_list, final_result_root):
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    plt.cla()
    sns.set_palette(palette)
    fontsize = 20
    fig, ax = plt.subplots(figsize=[8, 8])
    font = {'family': 'DejaVu Sans', 'size': fontsize}
    matplotlib.rc('font', **font)
    # sns.axes_style({'font.family': ['DejaVu Sans']})
    # fig.title('ROC')
    sns.stripplot()
    # 子图 ,
    axins = zoomed_inset_axes(ax, zoom=2.3, loc=4,borderpad=1)
    x1, x2, y1, y2 = 0.4, 0.6, 0.6, 0.8
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    #正图
    majorLocator = MultipleLocator(0.1)  # 将y轴主刻度标签设置为0.1的倍数
    majorFormatter = FormatStrFormatter('%1.1f')  # 设置y轴标签文本的格式
    axins.xaxis.set_major_locator(majorLocator)
    axins.yaxis.set_major_locator(majorLocator)
    ax.xaxis.set_major_formatter(majorFormatter)
    ax.yaxis.set_major_formatter(majorFormatter)
    for roc,model_name in zip(roc_list,model_list):
        fpr, tpr, au_roc = roc[0],roc[1],roc[2]
        ax.plot(fpr, tpr,
                 label='AUC {} : {}' .format(model_name,round(au_roc,3),
                 font = font))
        axins.plot(fpr, tpr, ls='-')
        ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        # fig.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
        #                 mode="expand", borderaxespad=0, ncol=3, fontsize=fontsize)
    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.7")
    ax.plot([0, 1], [0, 1], 'r--')
    ax.tick_params(labelsize=20)
    axins.tick_params(labelsize=10)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    # print labels
    [label.set_fontname('DejaVu Sans') for label in labels]

    ax.set_ylabel('True Positive Rate',fontdict=font)
    ax.set_xlabel('False Positive Rate',fontdict=font)
    image_name = os.path.join(final_result_root,"ROC_curve")
    if not os.path.exists(os.path.dirname(image_name)):
        os.mkdir(os.path.dirname(image_name))
    print("=> saving {}".format(image_name))
    plt.savefig(image_name,bbox_inches='tight')


def Draw(data_raw,axes,i,color,name):
    data = data_raw
    from scipy import stats
    bins = int(data.shape[0]/50)
    sns.distplot(data,
                 bins=bins,
                 fit =stats.gamma,
                 kde=False,
#                  rug=True,
                 color=color,
                 ax=axes[i])
#     axes[i].set_title(name,fontsize=23)
    axes[i].tick_params(labelsize=40)
    labels = axes[i].get_xticklabels() + axes[i].get_yticklabels()
    [label.set_fontname('DejaVu Sans') for label in labels]
#     axes[i].set_xlabel("Value")
#     axes[i].set_ylabel('probability')
    axes[i].legend(prop={'size': 12})  #设置图例
    axes[i].spines['top'].set_visible(False) #去掉上边框
    axes[i].spines['right'].set_visible(False) #去掉右边框
    axes[i].spines['left'].set_visible(False)  # 去掉左

def Feature_Comparison(img_root, hot_data,cold_data,feature):
    color = ["r","b"]
    f, axes = plt.subplots(2, 1, sharey=True,sharex=True, figsize=(20, 10))
    plt.cla()
    Draw(hot_data,axes,0,color[0],"Hotspots")
    Draw(cold_data,axes,1,color[1],"Coldspots")
    image_name = os.path.join(img_root, "{}.png".format(feature))
    plt.savefig(image_name)
    print("=>  {} SAVED".format(image_name))




