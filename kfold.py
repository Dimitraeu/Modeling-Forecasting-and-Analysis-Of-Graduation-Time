# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 12:13:28 2018

@author: Dimitra-Spuridoula
"""



#========================================================================================           
#aythh klash dhmioyrghtike gia na proseggizei ena stoixeio pou den uparxei
# h dhmiourghtite ekeinh thn stigmh
class Tree(dict):
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value
#========================================================================================           
class mykfold:
    def __init__(self,x,nfolds=5):
        self.__nfolds=nfolds
        self.__nData=x.shape[0]
         #edw ginetai h praksh gia na vgalei se kathe temaxio posa dedomena pane
        self.__nrest=(self.__nData)%self.__nfolds
        #edw ginetai h diairesh tou plithos  me to 5 kai to kanoume gia to dedomena elegxou se akeraia morfh
        self.__nDataTst=int(self.__nData/self.__nfolds)
        
        '''edw kanei afairesh ta dedomena test apo to plithos 
        twn dedomenwn kai vgainoun posa dedomena ekpaideyshs exoume
        '''
        self.__nDataTrn=self.__nData-self.__nDataTst
        '''
        to index xrhsimopoieitai na metatrepsei to arithimitko antikeimeno
        se akeraio arithmo to kanoume gia ola ta dedomena 
        '''
        self.__index_original=[i for i in range(self.__nData)]
        self.__index_fold=Tree()
        self.__index_fold['nfolds']=self.__nfolds
        self.__index_fold[0]['trn']=self.__index_original[:(self.__nDataTrn-self.__nrest)]
        self.__index_fold[0]['tst']=self.__index_original[(self.__nDataTrn-self.__nrest):]

        self.__make_folds()
        self.__validate()
        
    #edw ginetai peristrofh ths listas  
    def __rotate_list(self,L,n):
        return L[n:] + L[:n]
        
 #edw dhmiourgei thn ptyxh gia na mporoume na kanoume peristrofh ston pinaka   
    def __make_folds(self):
        L=self.__index_original
        for i in range(1,self.__nfolds):
            L=self.__rotate_list(L,self.__nDataTst)
            self.__index_fold[i]['trn']=L[:self.__nDataTrn]
            self.__index_fold[i]['tst']=L[self.__nDataTrn:]
       
#pairnei dedomena ta opoia kai exoun xwristei ta train kai se test
    def get_data_fold(self,foldid,x,y):
        itrn=self.__index_fold[foldid]['trn']
        itst=self.__index_fold[foldid]['tst']
        x_trn,x_tst=x[itrn],x[itst]
        y_trn,y_tst=y[itrn],y[itst]
        return x_trn,y_trn,x_tst,y_tst
   #ayth h synarthsh vazei tous deiktes stis metavlites 
    def get_index_fold(self,foldid):
        itrn=self.__index_fold[foldid]['trn']
        itst=self.__index_fold[foldid]['tst']
        return itrn,itst
        
  
    def __validate(self):
        sum=0
        for i in range(self.__index_fold['nfolds']):
            sum+=len(self.__index_fold[i]['tst'])
        assert(sum==self.__nData)#xrhsimopoieitai h assert gia ton entopismo sfalamtwn  
          
#exoume prosvash sto mhkos enos  pinaka metraei ta dedomena poy exei kathe kfold           
    def fold_lengths(self):
        sum=0
        fl=[]
        for i in range(self.__index_fold['nfolds']):
            l=len(self.__index_fold[i]['tst'])
            sum+=l
            fl.append(i)
        return fl

#========================================================================================        