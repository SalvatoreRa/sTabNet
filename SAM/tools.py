from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
import pandas as pd
import seaborn as sns
import gseapy as gp
from scipy.stats import t
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def keras_cat(target, categories):
    """ 
    take target variable and transforme it in array for keras
    perform one-hot encoding
    target = target variables
    categories = the categories available in the target
    output = an np.array, each column a categories, 0-1 for each categories
    """
    target = target.tolist()
    mapping = {}
    for x in range(len(categories)):
      mapping[categories[x]] = x
    for x in range(len(target)):
      target[x] = mapping[target[x]]
    one_hot_encode = to_categorical(target)
    return one_hot_encode



def history_plot(hist = None, cm_m = True, mod = None, test = None, _y =  None):
    '''
    plot history loss, accuracy and confusion matrix (optional)
    required
        history = keras history
    optional
        keras model = classifier
        test = test dataset
        y encoded = the encoded y 
    example usage
    #history_plot(hist = history, mod = model_go, cm_m = True, test = X_test,
                  _y = y_test_enc )
    #history_plot(history, True, model_go,  X_test,  y_test_enc )
    '''
    print(hist.history.keys())
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    plt.show()
    # summarize history for loss
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    
    if cm_m:
        #optional
        _preds = mod.predict(test)
        _y_preds = np.argmax(_preds, axis = 1)
        cm = confusion_matrix(_y[:,1], _y_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()

        

def recovery_weight(_layer  = None, annot = False, path_mat= None):
    '''
    recovery_weight v2
    take a linera_go layer as input and it is recovery the weights
    input
    take the pathway matrix the linear_layer 
    return
    the weight, the bias and the sparse matrix
    optional
    if annot true, return the annotation of the matrix in gather format
    example usage:
    w, b, wm = recovery_weight(_layer  = linear_layer, annot = True, path_mat= go )
    '''
    _w = _layer.weights[0].numpy()
    _b = _layer.weights[1].numpy()
    
    if annot:
        print("annotated matrix/ True")
        _b = pd.Series(_b, index = path_mat.columns)
        sparse_mat = tf.convert_to_tensor(path_mat, dtype=tf.float32)
        idx_sparse = tf.where(tf.not_equal(sparse_mat, 0))
        sparse = tf.SparseTensor(idx_sparse, _w, sparse_mat.get_shape())
        _wm = tf.sparse.to_dense(sparse).numpy()
        _wm = pd.DataFrame(_wm, index = path_mat.index,  columns = path_mat.columns).T
        _wm = pd.melt(_wm.reset_index(), id_vars='index')
        _wm = _wm[_wm.value != 0]
        _wm.columns =  ["to_node", "from_node", "weight_value"]
        
    return _w, _b, _wm

def list_avg_gene(_dir = None, _file = None, _n =20, y= 'genes'):
    '''
    return the plot of the top20 in absolute value, the csv with the average
    INPUT
    _dir = where to load and to save the file
    _file = the file where are stored data from a loop
    _N = numbero of genes to plot
    y = is for the column names for plotting, default is "genes" for genes
    OUTPUT
    bar plot
    csv with genes and average
    EXAMPLE USAGE
    list_avg_gene(_file_dir = "/home/sraieli/test/go_test/", 
              _file = "goconcat_net2_genes_mut.csv", _n =20)
    list_avg_gene(_dir =  "/home/sraieli/test/path_test/", 
              _file = "goconcat_net4_path.csv", _n =20, y = "pathway")
    '''
    _df = pd.read_csv(_dir + _file,  index_col=0)
    _df1 = _df.dropna(thresh=2)
    _df1["avg"] = _df1.iloc[:,1:_df1.shape[1]].mean(axis= 1)
    _df1 =_df1.reindex(_df1.avg.abs().sort_values(ascending=False).index)
    _df2 = _df1.iloc[0:_n,:].copy()
    ax = sns.barplot(data = _df2, x = 'avg', y = y)
    fig = ax.get_figure()
    _f = _file.split(".csv")
    fig.savefig(_dir +_f[0] +"_top_" + str(_n) + ".png", dpi = 300, bbox_inches='tight') 
    plt.close()
    _df2 = _df1[[y, "avg"]]
    _df2.to_csv(_dir +_f[0] + "_avg.csv")

def plot_weights(weights = None, n = 10):
    '''
    VERSION 2
    plot the first weights
    take a normalize matrix from recovery weight and plot a bar plot with from node
    input as column and output node in the center
    example usage
    w, b, wm = recovery_weight(_layer  = linear_layer, annot = True, path_mat= go )
    plot_weights(weights = wm, n = 10)
    
    '''
    
    weights = weights.sort_values(by=['weight_value'], ascending=False).iloc[0:n,:]
    weights["Input"] = [str(i) for i in range(n)]
    fig, ax = plt.subplots()    
    ax = sns.barplot(y = "Input", x= "weight_value", data = weights)
    ax.set_yticklabels(weights['from_node'])
    #ind = np.arange(len( weights["weight_value"]))
    

    k =weights["to_node"].to_list()
    for bar, i in zip(ax.patches, k[::-1]):
        ax.text(0, bar.get_y()+bar.get_height()/2,  i, color = 'black', ha = 'left', va = 'center')
        
def weight_means_plot(norm_w = None, n =10):
    '''
    plotting normalize by mean weights
    the most important by absolute value
    plot the input node
    EXAMPLE USAGE
    w, b, wm = recovery_weight(_layer  = linear_layer, annot = True)
    weight_means_plot(norm_w = wm, n =10)
    '''
    _w = norm_w.sort_values(by=['weight_value'], ascending=False)
    df =  _w .groupby(['from_node']).agg(['mean', 'count'])
    df.columns = [ '_'.join(str(i) for i in col) for col in df.columns]
    df.reset_index( inplace=True)
    df =df.reindex(df.weight_value_mean.abs(ascending=False).sort_values().index)
    df = df.iloc[0:n,:].sort_values(by=['weight_value_mean'], ascending=False)
    sns.barplot(data = df, x = 'weight_value_mean', y =  'from_node')

def plot_gene(genes_df= None, _y = None, gene = None):
    '''
    Input, take a df with genes, a category dataframe, the gene name
    EXAMPLE USAGE
    plot_gene(genes_df= df2, _y = outcome, gene = "MYC")
    '''
    cats = _y.iloc[:,0].value_counts().index.to_list()
    dfx = pd.DataFrame(genes_df[gene])
    dfx["cat_var"] = outcome.to_numpy().ravel()
    
    sns.displot(data =dfx, x = gene, hue = "cat_var",  kind="kde")
    plt.axvline(x=dfx[dfx["cat_var"]== cats[1]].iloc[:,0].mean(),color='lightblue', ls = '--')
    plt.axvline(x=dfx[dfx["cat_var"]== cats[0]].iloc[:,0].mean(),color='orange', ls = '--')



def enriching_list(gene_list = None, gene_col = 0, value_col=1, n = 100, 
                      method_enr = None):
    ''' take a data frame and return the top genes list
    INPUT
    gene_list = a data frame with genes and a value
    gene_col = default 0, the index of the col containing the genes names
    value_col = default 1, the index of the col containing the value for sorting
    n = default 100, number of genes to include in the list
    OPTIONAL INPUT
    method decide the sorting, available choices: None, "abs", "pos", "neg"
        abs use the absolute value, pos and neg ascending/descending
        None do not sort
    EXAMPLE USAGE
    from a linear layer:
    a = Linear_data( layer = linear_layer, path_mat = go)
    wm = a.weight_means()
    enr_list = enriching_list(gene_list = wm, gene_col = 0, value_col=1, n = 100, 
                              method_enr = "neg")
    
    from a df:
    gene_dir = "/home/sraieli/py_script/"
    nms = pd.read_csv(gene_dir + "vanilla_genes.csv", index_col=0)  
    nms = nms[["genes", "avg"]]
    enr_list =enriching_list(gene_list = rnk, gene_col = 0, value_col=1, n = 100, 
                              method_enr = "pos")
    
    '''
    if method_enr== None:
        gene_list = gene_list.iloc[0:n,:]
    
    if method_enr== "abs":
        gene_list =gene_list.reindex(gene_list.iloc[:,value_col].abs().sort_values(
        ascending=False).index)
        gene_list = gene_list.iloc[0:n,:]
    
    if method_enr== "pos":
        gene_list =gene_list.reindex(gene_list.iloc[:,value_col].sort_values(
        ascending=False).index)
        gene_list = gene_list.iloc[0:n,:]
        
    if method_enr== "neg":
        gene_list =gene_list.reindex(gene_list.iloc[:,value_col].sort_values(
        ascending=True).index)
        gene_list = gene_list.iloc[0:n,:]
    
    glist = gene_list.iloc[:,gene_col].squeeze().str.strip().tolist()
    
    return glist



def enrich_pathway(glist = None, geneset = 'KEGG_2016', saveout = False, 
                   out_dir = None, title = None, plot = False, n = 20):
    '''
    return enrichment
    INPUT
    glist = list of gene
    geneset = geneset available in enrichr library
    
    OTHER INFO
    to check available genesets
    names = gp.get_library_name()
    print(names)
    for custom gmt check the documentation
    
    OPTIONAL
    out_dir = allows to save in a csv
    plot = plot a dot plot
    example usage
    
    EXAMPLE USAGE
    out_dir = "/home/sraieli/test/results/"
    title = "go_test"
    enr_p = enrich_pathway(glist = enr_list, geneset = 'GO_Biological_Process_2015', saveout = True,
                       out_dir = out_dir, title = title, plot = True)
    
    '''
    enr = gp.enrichr(gene_list=glist,
                 gene_sets=geneset,
                 organism='Human', # don't forget to set organism to the one you desired! e.g. Yeast
                 cutoff=0.5 # test dataset, use lower value from range(0,1)
                )
    enr_csv = enr.results
    if saveout:
        print(f"enr csv saved in dir:{out_dir}")
        enr_csv.to_csv(out_dir+title+".csv")
    if plot:
        enric = enr_csv.iloc[0: n, :].copy()
        enric[['Nfound', 'Ntotal']] = enric['Overlap'].str.split('/', 1, expand=True)
        enric["percent_gene"] = enric['Nfound'].astype(int) /enric['Ntotal'].astype(int)
        ax = sns.relplot(data = enric, x = 'Adjusted P-value', y = 'Term', size="Combined Score",
            sizes=(40, 400), hue="percent_gene" )
        fig = ax.fig
        fig.savefig(out_dir+title + ".png", dpi = 300, bbox_inches='tight') 
    return enr_csv


def compute_corrected_ttest(serie1, serie2, dsize = [144,40]):
    """Computes right-tailed paired t-test with corrected variance
    using Corrects standard deviation using Nadeau and Bengio's approach
    serie1 = ndarray of shape or list
    serie2 = ndarray of shape or list
    dsize = size of train and test et
    dsize 1= y_train size, for calculate number of observation
    dsize 2 = y_test size, for calculate number of observation
    https://scikit-learn.org/0.24/auto_examples/model_selection/plot_grid_search_stats.html
    """
    differences = [y - x for y, x in zip(serie1, serie2)]
    n_train =dsize[0]
    n_test =dsize[1]
    n = n_train + n_test
    df = n -1
    
    corrected_var = (
        np.var(differences, ddof=1) * ((1 / n) + (n_test / n_train))
    )
    corrected_std = np.sqrt(corrected_var)
    
    mean = np.mean(differences)
    t_stat = mean / corrected_std
    p_val = t.sf(np.abs(t_stat), df)  # right-tailed t-test
    return t_stat, p_val

def compare_nets(res_dir = None, file1 = None, file2 = None, dsize = [0,0], saving = False, 
                 title = None, printing = False):
    '''
    compare two network, plot and return Benjo t_test comparison
    INPUT
    res_dir = where the csv file are 
    file1 = first csv file. results of a model
    file2 = second csv file. results of a model
    dsize = dataset dimension, list (first number train size, second number test size)
    OUTPUT
    plot of the evaluation metric compared
    result of statistical comparison, a dataframe
    OPTIONAL
    title = if you want to save, it will use the title
    printing = print the statistical comparison
    EXAMPLE USAGE
    records =compare_nets(res_dir = "/home/sraieli/test/go_test/", file1 = "goconcat_results_net.csv", 
                 file2 = "go_results_net.csv", dsize = [144, 40], saving = True, 
                 title = "go_net_comparison", printing = False)
    '''
    _df = pd.read_csv(res_dir + file1,  index_col=0)
    _df1 = pd.read_csv(res_dir + file2,  index_col=0)

    fig, axes = plt.subplots(3, 4, figsize=(20, 20))
    fig.subplots_adjust(hspace = .5, wspace=0.2)
    axs = axes.ravel()

    for i in range(_df.shape[1]):
        sns.histplot(_df1[_df1.columns[i]], ax = axs[i], kde=True )
        axs[i].axvline(x=_df1[_df1.columns[i]].mean(),color='red', linestyle = "--")
        sns.histplot(_df[_df.columns[i]], ax = axs[i], kde=True, color ="green" )
        axs[i].axvline(x=_df[_df.columns[i]].mean(),color='red', linestyle = "--")
    cols = ["eval_measure", "t_stat", "p_val", "mean_net1", "mean_net2", "net1", "net2"]
    record = pd.DataFrame(index = range(_df.shape[1]), columns = cols)
    
    for i in range(_df.shape[1]) :
        t_stat, p_val = compute_corrected_ttest(_df.iloc[:, i], _df1.iloc[:, i],
                                        dsize = dsize)
        m = _df.columns[i]
        mv = _df.iloc[:, i].mean()
        mr = _df1.iloc[:, i].mean()
        record.iloc[i,0:8] = m, t_stat, p_val, mv, mr,file1, file2 
        if printing:
            print(f" m: {m}  t-value: {t_stat:.3f} --Cor. p-value: {p_val:.3f}--\n"
             f"net1 mean: {mv:.3f} and net2 mean: {mr:.3f}")
        
    if saving:
        plt.savefig(res_dir + title + ".jpeg")
        record.to_csv(res_dir + title + ".csv")    
    
    
    return record


def list_avg_gene(_dir = None, _file = None, _n =20, y= 'genes'):
    '''
    return the plot of the top20 in absolute value, the csv with the average
    INPUT
    _dir = where to load and to save the file
    _file = the file where are stored data from a loop
    _N = numbero of genes to plot
    y = is for the column names for plotting, default is "genes" for genes
    OUTPUT
    bar plot
    csv with genes and average
    EXAMPLE USAGE
    list_avg_gene(_file_dir = "/home/sraieli/test/go_test/", 
              _file = "goconcat_net2_genes_mut.csv", _n =20)
    list_avg_gene(_dir =  "/home/sraieli/test/path_test/", 
              _file = "goconcat_net4_path.csv", _n =20, y = "pathway")
    '''
    _df = pd.read_csv(_dir + _file,  index_col=0)
    _df1 = _df.dropna(thresh=2)
    _df1["avg"] = _df1.iloc[:,1:_df1.shape[1]].mean(axis= 1)
    _df1 =_df1.reindex(_df1.avg.abs().sort_values(ascending=False).index)
    _df2 = _df1.iloc[0:_n,:].copy()
    ax = sns.barplot(data = _df2, x = 'avg', y = y)
    fig = ax.get_figure()
    _f = _file.split(".csv")
    fig.savefig(_dir +_f[0] +"_top_" + str(_n) + ".png", dpi = 300, bbox_inches='tight') 
    plt.close()
    _df2 = _df1[[y, "avg"]]
    _df2.to_csv(_dir +_f[0] + "_avg.csv")
    
    

def prerank_enrich(gene_dir = None, file1 = None, title = None, geneset = 'KEGG_2016', perm = 100):
    '''
    GSEA pre-ranked, it creates a new directory with all the file
    caution it can kill the kernel more than 100 perm
    EXAMPLE USAGE
    enr = prerank_enrich(gene_dir = "/home/sraieli/test/go_test/", file1 = 'goconcat_net2_genes_avg.csv', 
                     title = 'gonet2_genes_prerank',
               geneset = 'GO_Biological_Process_2015', perm = 100)
    
    '''
    path = gene_dir + title
    try: 
        os.mkdir(path) 
    except OSError as error: 
        print(error)  
    nms = pd.read_csv(gene_dir + file1, index_col=0)
    pre_res = gp.prerank(rnk=nms, gene_sets=geneset,
                     processes=32,
                     permutation_num=perm, # reduce number to speed up testing
                     outdir= path, format='png', seed=6)

    enrich = pre_res.res2d.sort_index()
    n = 20
    enric = enrich.sort_values("pval", ascending=True).copy()
    enric["percent_gene"] = enric['matched_size'].astype(int) /enric['geneset_size'].astype(int)
    enric["pval"] = np.where( enric["pval"] == 0, 1/perm, enric["pval"])
    enric["-log_pval"] = -np.log(enric["pval"])
    enrtot = enric.iloc[0: n, :].copy()
    ax = sns.relplot(data = enrtot, x = '-log_pval', y = 'Term', size="percent_gene",
                sizes=(40, 400), hue="nes" )
    fig = ax.fig
    fig.savefig(path + "/_top20pathway.png", dpi = 300, bbox_inches='tight') 
    plt.close()

    enrneg = enric.sort_values("nes", ascending=True).iloc[0: n, :].sort_values("-log_pval", ascending=False)
    ax = sns.relplot(data = enrneg, x = '-log_pval', y = 'Term', size="percent_gene",
                sizes=(40, 400), hue="nes", col_order=enrneg.index.tolist() )
    fig = ax.fig
    fig.savefig(path + "/_top20neg_pathway.png", dpi = 300, bbox_inches='tight') 
    plt.close()

    enrpos = enric.sort_values("nes", ascending=False).iloc[0: n, :].sort_values("-log_pval", ascending=False)
    ax = sns.relplot(data = enrpos, x = '-log_pval', y = 'Term', size="percent_gene",
                sizes=(40, 400), hue="nes", col_order=enrneg.index.tolist() )

    fig = ax.fig
    fig.savefig(path + "/_top20pos_pathway.png", dpi = 300, bbox_inches='tight') 
    plt.close()
    return enrich


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def splitting_data( data_x = None, data_y = None, val_split = 0.1, test_split = 0.2,
                  random_seed =42):

    """ splitting dataset in train, val, test
    this function take a dataset and it is splitting in three parts
    data are returning scaled (data leakage is avoided)
    category variable are returne encoded
    this is for classification dataset
    INPUT
    data_x = matrix (or dataframe), this the X dataset, a matrix
    data_y = a panda df (with only one series, or it can be a series)
    random_seed = the random seed, default is 42
    val_split = percentage of split of validation set , default = 0.1
    test_split = percentage of split of test set , default = 0.2
    OUTPUT
    _X_train, _X_test, _X_val = splitted data_x, data are independently scaled
    _y_train_enc, _y_val_enc, _y_test_enc = encoded y variable
    EXAMPLE OF USAGE
    X_train, X_test, X_val, y_train_enc, y_val_enc, y_test_enc =splitting_data( data_x = concat, 
                                                                            data_y = outcome, 
                                                                            val_split = 0.1, 
                                                                            test_split = 0.2,
                                                                            random_seed =42)
    
    """
    
    _X_train, _X_test, _y_train, _y_test = train_test_split(data_x, data_y, 
                                                            test_size=test_split, 
                                                            random_state=random_seed, 
                                                            stratify = data_y)
    _X_train, _X_val, _y_train, _y_val = train_test_split(_X_train, _y_train, 
                                                          test_size=val_split, 
                                                          random_state=random_seed, 
                                                          stratify = _y_train)
    scaler = MinMaxScaler()
    scaler.fit(_X_train)
    _X_train = scaler.transform(_X_train)
    _X_val = scaler.transform(_X_val)
    _X_test = scaler.transform(_X_test)
    try:
        tar_categ = data_y.iloc[:,0].value_counts().index.tolist()
        _y_train_enc = keras_cat( _y_train.iloc[:,0], tar_categ)
        _y_val_enc = keras_cat( _y_val.iloc[:,0],tar_categ)
        _y_test_enc = keras_cat( _y_test.iloc[:,0],tar_categ)
    except:
        print('pd.series detected')
        tar_categ = False
    finally:
        if tar_categ == False:
            tar_categ = data_y.value_counts().index.tolist()
            _y_train_enc = keras_cat( _y_train, tar_categ)
            _y_val_enc = keras_cat( _y_val, tar_categ)
            _y_test_enc = keras_cat( _y_test, tar_categ)
    
    return _X_train, _X_test, _X_val, _y_train_enc, _y_val_enc, _y_test_enc
    
    


def loop_go_conc1(_model = None, _path_mat = None, l = 2, w_norm = "sigma_5"):
        ''' 
        return the gene weights of the gene layer for the loop
        INPUT
        _model = a model GO model
        _path_mat = pathway matrix
        l = number of the layer in the model, the gene layer (a linearGO layer)
        w_norm = normalization method, sigma5 is the default
        OUTPUT
        normalized gene weights
        EXAMPLE USAGE
        gene =loop_go_conc1(_model = model_go, _path_mat = go2, l = 2, w_norm = "sigma_5")
        '''
        _layer = _model.layers[l] 
        _w = _layer.weights[0].numpy()
        sparse_mat = tf.convert_to_tensor(_path_mat, dtype=tf.float32)
        idx_sparse = tf.where(tf.not_equal(sparse_mat, 0))
        sparse = tf.SparseTensor(idx_sparse, _w, sparse_mat.get_shape())
        _wm = tf.sparse.to_dense(sparse).numpy()
        _wm = pd.DataFrame(_wm, index = _path_mat.index,  
                                columns = _path_mat.columns).T
        _wm = pd.melt(_wm.reset_index(), id_vars='index')
        _wm = _wm[_wm.value != 0]
        _wm.columns =  ["to_node", "from_node", "weight_value"]

        _wm = _wm.sort_values(by=['weight_value'], ascending=False)
        _wm= _wm.sort_values(by=['weight_value'], ascending=False)
        _df =  _wm.groupby(['from_node']).agg(['mean', 'count'])
        _df.columns = [ '_'.join(str(i) for i in col) for col in _df.columns]
        _df.reset_index( inplace=True)
        _df =_df.reindex(_df.weight_value_mean.abs().sort_values(ascending=False).index)

        if w_norm == "sigma_5":
                    _wm= _wm.sort_values(by=['weight_value'], ascending=False)
                    _df =  _wm .groupby(['from_node']).agg(['sum', 'count'])
                    _df.columns = [ '_'.join(str(i) for i in col) for col in _df.columns]
                    _df.reset_index( inplace=True)
                    _df =_df.reindex(_df.weight_value_sum.abs().sort_values(ascending=False).index)
                    sigma_5 = np.std(_df.to_node_count) * 5
                    _df["weight_value_mean"] =np.where(_df.iloc[:,2] >= sigma_5, _df.iloc[:,3]/_df.iloc[:,2],
                                                     _df.iloc[:,3])
                    _df = _df[["from_node", "weight_value_mean", "weight_value_count"]]

        _df = _df.iloc[:,0:2] #gene value
        return _df
    

def loop_go_conc3(_model = None, _conc_mat = None, l = 1):
        ''' easy returning of data for loop for concat
            #block component
            weight from mutation and gene level
            INPUT
            _model = a model GO model
            _conc_mat = concat matrix
            l = number of the layer in the model, the gene layer (a linearGO layer)
            
            OUTPUT
            normalized gene weights
            
            EXAMPLE USAGE
            gene_mut = loop_go_conc3(_model = model_go, _conc_mat = adj_gene, l = 1)
        '''
        _layer = _model.layers[l]
        _w = _layer.weights[0].numpy() 

        sparse_mat = tf.convert_to_tensor(_conc_mat, dtype=tf.float32)
        idx_sparse = tf.where(tf.not_equal(sparse_mat, 0))
        sparse = tf.SparseTensor(idx_sparse, _w, sparse_mat.get_shape())
        _wm = tf.sparse.to_dense(sparse).numpy()
        _wm = pd.DataFrame(_wm, index = _conc_mat.index,  
                                   columns = _conc_mat.columns).T
        _wm = _wm.reset_index()
        _wm = pd.melt(_wm, id_vars='index')
        _wm = _wm[_wm.value != 0]
        _wm.columns =  ["to_node", "from_node", "weight_value"]

        _wm = _wm.sort_values(by=['weight_value'], ascending=False)
        _df =  _wm .groupby(['from_node']).agg(['mean', 'count'])
        _df.columns = [ '_'.join(str(i) for i in col) for col in _df.columns]
        _df.reset_index( inplace=True)
        _df =_df.reindex(_df.weight_value_mean.abs().sort_values(ascending=False).index)
        _df = _df.iloc[:,0:2] #block component
        return _df

def loop_go_conc4(_model = None, _path_mat = None, l = 3):
        ''' easy returning of data for loop for concat
            #pathway from layer 
            return the gene weights of the pathway layer for the loop
            INPUT
            _model = a model GO model
            _path_mat = pathway matrix
            l = number of the layer in the model, the pathway layer (a linearGO layer)
            
            OUTPUT
            normalized gene weights
            EXAMPLE USAGE
            path =loop_go_conc4(_model = model_go, _path_mat = go2, l = 3)
        '''
        _layer = _model.layers[l]
        _wm = _layer.weights[0].numpy() 
        _wm =pd.DataFrame(_wm, index = _path_mat.columns)

        _wm["avg"] = _wm.iloc[:,0:_wm.shape[1]].mean(axis= 1)
        _wm =_wm.reindex(_wm.avg.abs().sort_values(ascending=False).index)

        _wm["pathway"] = _wm.index
        _wm = _wm[["pathway", "avg"]]
        
        return _wm

def list_avg_path(_dir = None, _file = None, _path_mat =None, y= 'pathway', go = True):
    '''
    return the csv with the average of the path,
    INPUT
    _dir = where to load and to save the file
    _file = the file where are stored data from a loop
    y = is for the column names for plotting, default is "pathway" for pathway
    _path_mat = matrix for the pathway for the normalization
    OUTPUT
    annotated csv, with normalization sigma 3
    OPTIONAL
    go = True, default
    if they are gene ontology pathway it gives back the original name
    EXAMPLE USAGE
    list_avg_path(_dir = "/home/sraieli/test/path_test/", _file = "goconcat_net4_path.csv", 
              _path_mat =go2, y= 'pathway', go = True)
    '''
    _df = pd.read_csv(_dir + _file,  index_col=0)
    _df1 = _df.dropna(thresh=2)
    _df1["avg"] = _df1.iloc[:,1:_df1.shape[1]].mean(axis= 1)
    _df1 =_df1.reindex(_df1.avg.abs().sort_values(ascending=False).index)
    
    _path_mat = _path_mat.transpose()
    _df1["avg"] = _df1.iloc[:,1:_df1.shape[1]].mean(axis= 1)
    _path_mat["sum_gene"] = _path_mat.sum(axis= 1)
    _path_mat = _path_mat[_path_mat.index.isin(_df1[y])]
    _path_mat = _path_mat.reset_index(level=0)
    _path_mat = _path_mat[["index","sum_gene"]]
    sigma_3 = np.std(_path_mat.sum_gene) * 3

    _df2 = _df1[[y, "avg"]]
    _df2 = _df2.merge(_path_mat, how = "left",left_on = "pathway", right_on = "index")
    _df2 = _df2.drop("index", axis =1)
    _df2["norm_sigma3"] = np.where(_df2.iloc[:,2] >= sigma_3, _df2.iloc[:,1]/_df2.iloc[:,2],
                                                         _df2.iloc[:,1])
    if go:
        my_dir = "/home/sraieli/py_script/pnet/"
        f = "gonames.txt"
        g = pd.read_csv(my_dir + f,   sep="\t", header=None)
        g.columns =["pathway", "path_name"]                
        _df2 = _df2.merge(g, how = "left", left_on = "pathway", right_on = "pathway") 
        
    _f = _file.split(".csv")
    _df2 = _df2.sort_values(by = "avg")
    _df2.to_csv(_dir +_f[0] + "_avg_norm.csv")
    
def list_avg_path2(_dir = None, _file = None, _path_mat =None, y= 'pathway', go_op = True):
    '''
    return the csv with the average of the path,
    INPUT
    _dir = where to load and to save the file
    _file = the file where are stored data from a loop
    y = is for the column names for plotting, default is "pathway" for pathway
    _path_mat = matrix for the pathway for the normalization
    OUTPUT
    annotated csv, with normalization sigma 3
    OPTIONAL
    go = True, default
    if they are gene ontology pathway it gives back the original name
    COLUMN EXPLICATION
    ratio = ratio number genes after filtering/original number of gene in the pathway
    sigma3 = pathway with more genes than 3 sigma got penalized
    norm_ratio = pathway normalized afer ratio
    EXAMPLE USAGE
    list_avg_path2(_dir = "/home/sraieli/test/path_test/", _file = "goconcat_net4_path.csv", 
              _path_mat =go2, y= 'pathway', go_op = True)
    '''
    _df = pd.read_csv(_dir + _file,  index_col=0)
    _df1 = _df.dropna(thresh=2)
    _df1["avg"] = _df1.iloc[:,1:_df1.shape[1]].mean(axis= 1)
    _df1 =_df1.reindex(_df1.avg.abs().sort_values(ascending=False).index)
    
    _path_mat = _path_mat.transpose()
    _df1["avg"] = _df1.iloc[:,1:_df1.shape[1]].mean(axis= 1)
    _path_mat["sum_gene"] = _path_mat.sum(axis= 1)
    _path_mat = _path_mat[_path_mat.index.isin(_df1[y])]
    _path_mat = _path_mat.reset_index(level=0)
    _path_mat = _path_mat[["index","sum_gene"]]
    sigma_3 = np.std(_path_mat.sum_gene) * 3

    _df2 = _df1[[y, "avg"]]
    _df2 = _df2.merge(_path_mat, how = "left",left_on = "pathway", right_on = "index")
    _df2 = _df2.drop("index", axis =1)
    _df2["norm_sigma3"] = np.where(_df2.iloc[:,2] >= sigma_3, _df2.iloc[:,1]/_df2.iloc[:,2],
                                                         _df2.iloc[:,1])
    if go_op:
        my_dir = "/home/sraieli/py_script/pnet/"
        f = "gonames.txt"
        f1 = "GOpath_matrix.txt"
        g = pd.read_csv(my_dir + f,   sep="\t", header=None)
        g.columns =["pathway", "path_name"]
        _df2 = _df2.merge(g, how = "left", left_on = "pathway", right_on = "pathway")

        #path gene matrix before filtering
        gox = pd.read_csv(my_dir + f1,   sep="\t")
        gox = gox.transpose().iloc[1:,:]
        gox["original_n"] = gox.sum(axis= 1)
        gox["ptw"] = gox.index
        gox = gox[["ptw","original_n"]]
        _df2 = _df2.merge(gox, how = "left",left_on = "pathway", right_on = "ptw")
        _df2 = _df2.drop("ptw", axis =1)
        _df2["ratio"] = _df2["sum_gene"] / _df2["original_n"]
        _df2["norm_ratio"] =_df2["norm_sigma3"] * _df2["ratio"]
    
    _f = _file.split(".csv")
    _df2 = _df2.sort_values(by = "avg")
    _df2.to_csv(_dir +_f[0] + "_avg_normx.csv")
    

def plot_result(res_dir = None, out_dir = None, file =None, title= None):
    '''
    plot results for a model
    plot the average and histogram of the different splits
    
    '''
    _df = pd.read_csv(res_dir + file,  index_col=0)
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 20))
    fig.subplots_adjust(hspace = .5, wspace=0.2)
    axs = axes.ravel()

    for i in range(_df.shape[1]):

        sns.histplot(_df[_df.columns[i]], ax = axs[i], kde=True, color ="green" )
        axs[i].axvline(x=_df[_df.columns[i]].mean(),color='red', linestyle = "--")
    
        
    plt.savefig(out_dir + title + ".jpeg")
