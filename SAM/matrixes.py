import pandas as pd
import numpy as np
import re

def genes_go_matrix(gene_df = pd.DataFrame(), gene_pat = pd.DataFrame(), filt_mat = None, 
                   min_max_gene = [10,100], pat_num = 1):
    '''
    function: this generate the first matrix genes-go pathway
    input:
    gene_df = a pandas df, with samples as row, columns are genes
    gene_pat = a pandas df, genes are row, column are pathway
    filt_mat = a pandas df with pathway as columns
    output:
    return gene_df and gene_pat
    gene_df is filtered for common genes
    gene_pat is filtered by genes in gene_df, pathway in the filt_mat
    pat_num = number of pathways a gene need to link
    EXAMPLE USAGE:
    df2, go2 = genes_go_matrix(gene_df = df1, gene_pat = go, filt_mat = first_matrix_connection, 
                   min_max_gene = [5,200], pat_num = 1)
    '''
    #filtrage of the matrix for P-Net hierarchical costruction
    if filt_mat is not None:
        col_list = [go.columns[0]] + filt_mat.columns.tolist()
        gene_pat = gene_pat.iloc[:, gene_pat.columns.isin(col_list)]
    #alphabetical order
    gene_df = gene_df.reindex(sorted(gene_df.columns), axis=1)
    gene_pat = gene_pat.reindex(sorted(gene_pat.columns), axis=1)
    #filter gene not in the expression matrix, filter also genes not in pathway
    gene_pat = gene_pat[ gene_pat["genes"].isin(gene_df.columns)]
    gene_df = gene_df.iloc[:, gene_df.columns.isin(gene_pat["genes"])]
    #filter pathway with a selected number of genes
    #filtering genes in a certain number of pathway
    gene_pat.index = gene_pat.genes.tolist()
    gene_pat =gene_pat.drop("genes", axis = 1)
    gene_pat = gene_pat.loc[:, (gene_pat.sum() >=min_max_gene[0]) & (gene_pat.sum() <=min_max_gene[1])]
    gene_pat = gene_pat.loc[gene_pat.sum(axis =1) >=pat_num, :]
    gene_df = gene_df.iloc[:, gene_df.columns.isin(gene_pat.index.tolist())]
    return gene_df, gene_pat

def go_add_matrix(go_mat = pd.DataFrame(), go_lvl =pd.DataFrame(), 
                  go_adj = pd.DataFrame(), lvl = 7, filt = 1):
    ''' 
    this generate a generic matrix pathway-pathway int
    input:
    go_mat = a pandas df, the previous gene-pathway matrix
            alternatively the result of a call go_add_matrix()
    go_lvl = pandas df, containing the levels of the pathway in go
    go_adj = a pandas df with adiancency matrix
    lvl = select the level you want to use
    filt (default 1) = minimun number connection
    output:
    return return a matrix df with a pathway adiancency interaction
    the rows of the adiacency matrix are filtered by go_mat
    the columns of the adiacency matrix are filtered by pathway in the selected level
    Notes: 
    * the low level have less pathway, higher in pathway hierarchy (ex. len(lvl7)>len(lvl6))
    * start from 6, you could not have connection otherwise
    '''
    go_mat = go_mat.reindex(sorted(go_mat.columns),  axis=1)
    go_lvl = go_lvl[go_lvl[str(lvl)] == 1].index
    go_adj = go_adj.iloc[go_adj.index.isin(go_mat.columns), go_adj.columns.isin(go_lvl)]
    go_adj = go_adj.reindex(sorted(go_adj.columns),  axis=1).sort_index()
    if filt is not None:
        go_adj = go_adj.loc[:, go_adj.sum() >= filt ]
    return go_adj


def concat_omics(_X = None, _mut = None, sort = False):
    ''' 
    return concatened omics matrix and and a connected matrix
    input 
    _x = rnaseq dataset
    _mut = mutation dataset
    output
    concat_matrix = concatatenation of the two dataset
    adiacency matrix = of the two omics and the gene (0,1 matrix)
    OPTIONAL
    sorting the concat matrix
    EXAMPLE USAGE
    X, y, go, mut = random_dataset_mutation(pat = 10, genes =10, pathway = 5, ratio = 0.5)
    concat, adj_gene = concat_omics(_X = X, _mut = mut, sort = True)

    '''
    _X = pd.concat([_X.reset_index(drop=True),_mut.reset_index(drop=True)], axis=1)
    df = pd.DataFrame()
    df["input_g"] =_X.columns
    df["genes"] =_X.columns
    df["connect"] = 1
    df["genes"] = [re.sub('\*$', '', df.iloc[s, 1]) for s in range(df.shape[0])]
    df = df.pivot_table(index="input_g", columns="genes", 
                     values="connect")
    df.columns.name = None
    df = df.reset_index().fillna(0)
    df.index = df.input_g.tolist()
    df = df.drop("input_g", axis = 1)
    if sort:
        _X = _X.reindex(sorted(_X.columns), axis=1)
    return _X, df


def concat_go_matrix(expr = None, exo_df = None, gene_pats =None, filt_mats = None, 
                   min_max_genes = [10,100], pat_nums = 1):
    '''
    return concatened omics matrix and and a connected matrix, the pathway matrix
    this function return a concatenated matrix of the two omics, a gene-go pathway
    check other function for more information
    EXAMPLE USAGE1
    X, y, go, mut = random_dataset_mutation(pat = 10, genes =10, pathway = 5, ratio = 0.5)
    concat, adj_gene, go2= concat_go_matrix(expr = X, exo_df = mut, gene_pats =go, filt_mats = None, 
                       min_max_genes = [10,100], pat_nums = 1)
    EXAMPLE USAGE2
    concat, adj_gene, go2= concat_go_matrix(expr = df1, exo_df = exome, gene_pats =go, filt_mats = None, 
                       min_max_genes = [10,100], pat_nums = 1)
    
    '''
    
    expr.columns = [re.sub('\*$', '', expr.columns[s]) for s in range(expr.shape[1])]
    expr, gene_pats = genes_go_matrix(gene_df = expr, gene_pat = gene_pats, filt_mat = filt_mats, 
                       min_max_gene = min_max_genes, pat_num = pat_nums)

    exo_df = exo_df.iloc[:, exo_df.columns.isin(gene_pats.index.tolist())]
    exo_df = exo_df.reindex(sorted(exo_df.columns), axis=1)
    expr.columns  = [i +"*" for i in expr.columns.tolist()]

    _concat, _adj_gene = concat_omics(_X = expr, _mut = exo_df, sort = True)
    return _concat, _adj_gene, gene_pats


