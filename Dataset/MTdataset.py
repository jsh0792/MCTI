import numpy as np
import pandas as pd
import torch
import os
from torch.utils.data import Dataset
from pathlib import Path
import glob

class MTDataset(Dataset):
    def __init__(self,
        csv_path = 'dataset_csv/ccrcc_clean.csv', data_dir=None,
        shuffle = False, seed = 7, print_info = True, n_bins = 4, ignore=[],
        patient_strat=False, label_col = None, eps=1e-6):

        super(MTDataset, self).__init__()
        self.custom_test_ids = None
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
        self.data_dir = data_dir

        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)

        self.patient_gene = torch.load('multitask/gene_data/normalized/{}.pt'.format(self.data_dir.split('/')[-1].lower()))
        
        slide_data = pd.read_csv(csv_path, low_memory=False)

        clinical_patient = slide_data['Patient ID']
        gene_patient = []
        for k in self.patient_gene:
            if k != 'gene':
                gene_patient.append(k[:-3])
        gene_patient = set(gene_patient)
        clinical_patient = set(clinical_patient)
        intersec_patient = clinical_patient & gene_patient
        
        slide_data = slide_data.loc[slide_data['Patient ID'].isin(intersec_patient)].reset_index()  
        tmp_set = set([item for item in slide_data['Patient ID'].values])       

        slide_data_existed = set([Path(file).stem[:12] for file in os.listdir(os.path.join(data_dir, 'pt_files'))]) 
        intersec_set = list(tmp_set & slide_data_existed)
        slide_data_new = intersec_set
        slide_data = slide_data.loc[slide_data['Patient ID'].isin(slide_data_new)].reset_index()   

        if not label_col:
            label_col = 'survival_months'
        else:
            assert label_col in slide_data.columns
        self.label_col = label_col

        patients_df = slide_data.copy()
        uncensored_df = patients_df[patients_df['censor'] < 1] 

        disc_labels, q_bins = pd.qcut(uncensored_df[label_col], q=n_bins, retbins=True, labels=False)
        q_bins[-1] = slide_data[label_col].max() + eps
        q_bins[0] = slide_data[label_col].min() - eps
        
        disc_labels, q_bins = pd.cut(patients_df[label_col], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        patients_df.insert(2, 'label', disc_labels.values.astype(int))

        patient_dict = {}
        for patient in patients_df['Patient ID']:
            slide_exp = os.path.join(data_dir, 'pt_files', patient) + '*'   

            slide_ids = glob.glob(slide_exp)
            
            patient_dict.update({patient:slide_ids})    
    
        self.patient_dict = patient_dict
    
        slide_data = patients_df
        slide_data.reset_index(drop=True, inplace=True)
        slide_data = slide_data.assign(slide_id=slide_data['Patient ID'])

        label_dict = {}     
        key_count = 0
        for i in range(len(q_bins)-1):
            for c in [0, 1]:
                print('{} : {}'.format((i, c), key_count))
                label_dict.update({(i, c):key_count})
                key_count+=1
        
#         level_0  index  label  ...  censor    survival_months      slide_id
# 0          1      1      3  ...       1          131.57           TCGA-3C-AALI
# 1          2      2      2  ...       1           48.42           TCGA-3C-AALJ

        self.label_dict = label_dict
        for i in slide_data.index:
            key = slide_data.loc[i, 'label']
            slide_data.at[i, 'disc_label'] = (int)(key)    
            censorship = slide_data.loc[i, 'censor']
            key = (key, int(censorship))
            slide_data.at[i, 'label'] = label_dict[key]     
#      level_0  index  label  Unnamed: 0  ... censor survival_months      slide_id       disc_label
# 0          1      1      7           1  ...      1          131.57    TCGA-3C-AALI         3.0
# 1          2      2      5           2  ...      1           48.42    TCGA-3C-AALJ         2.0

# 在临时增加后，但是cls_label类型应该是int，而不是float64.
#      level_0  index  label  Unnamed: 0  ... survival_months      slide_id  disc_label  cls_label
# 0          1      1      7           1  ...          131.57  TCGA-3C-AALI         3.0        7.0
# 1          2      2      5           2  ...           48.42  TCGA-3C-AALJ         2.0        5.0

        self.bins = q_bins
        self.num_classes=len(slide_data['cls_label'].value_counts())
        patients_df = slide_data.drop_duplicates(['Patient ID'])
        self.patient_data = {'Patient ID':patients_df['Patient ID'].values, 'label':patients_df['label'].values}

        self.slide_data = slide_data
        self.surv_discclass_num = len(patients_df['label'].value_counts())
        self.cls_ids_prep()
        self.get_gene_index()   
        


    def get_gene_index(self):   
        signatures = pd.read_csv('gene_data/signatures.csv')
        Tumor_Suppressor_Genes = signatures['Tumor Suppressor Genes'].dropna().to_list()
        Oncogenes = signatures['Oncogenes'].dropna().to_list()
        Protein_Kinases = signatures['Protein Kinases'].dropna().to_list()
        Cell_Differentiation_Markers = signatures['Cell Differentiation Markers'].dropna().to_list()
        Transcription_Factors = signatures['Transcription Factors'].dropna().to_list()
        Cytokines_and_Growth_Factors = signatures['Cytokines and Growth Factors'].dropna().to_list()

        self.indices0 = [index for index, value in enumerate(self.patient_gene['gene']) if value in Tumor_Suppressor_Genes]
        self.indices1 = [index for index, value in enumerate(self.patient_gene['gene']) if value in Oncogenes]
        self.indices2 = [index for index, value in enumerate(self.patient_gene['gene']) if value in Protein_Kinases]
        self.indices3 = [index for index, value in enumerate(self.patient_gene['gene']) if value in Cell_Differentiation_Markers]
        self.indices4 = [index for index, value in enumerate(self.patient_gene['gene']) if value in Transcription_Factors]
        self.indices5 = [index for index, value in enumerate(self.patient_gene['gene']) if value in Cytokines_and_Growth_Factors]
        

    def get_gene_num(self):
        return len(self.patient_gene['gene'])
    
    def get_class_num(self):
        return self.num_classes

    def cls_ids_prep(self):
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['cls_label'] == i)[0]
        
        self.slide_surv_ids = [[] for i in range(self.surv_discclass_num)]
        for i in range(self.surv_discclass_num):
            self.slide_surv_ids[i] = np.where(self.slide_data['label'] == i)[0]
    
    def get_split_from_df(self, all_splits: dict, split_key: str='train', scaler=None):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(split.tolist()) 
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = Generic_Split(df_slice, data_dir=self.data_dir, label_col=self.label_col, patient_dict=self.patient_dict, num_classes=self.num_classes, patient_gene=self.patient_gene)
        else:
            split = None
        
        return split

    def return_splits(self, from_id: bool=True, csv_path: str=None):
        if from_id:
            raise NotImplementedError
        else:
            assert csv_path 
            all_splits = pd.read_csv(csv_path)
            train_split = self.get_split_from_df(all_splits=all_splits, split_key='train')
            val_split = self.get_split_from_df(all_splits=all_splits, split_key='val')
            test_split = self.get_split_from_df(all_splits=all_splits, split_key='test')

        return train_split, val_split, test_split
    
    def getlabel(self, ids):
        return (self.slide_data['cls_label'][ids], self.slide_data['label'][ids])

    def __len__(self):
        return len(self.slide_data)

    def __getitem__(self, idx):
        case_id = self.slide_data['Patient ID'][idx]
        Y_surv = self.slide_data['disc_label'][idx]
        event_time = self.slide_data[self.label_col][idx]
        c = self.slide_data['censor'][idx]
        slide_ids = self.patient_dict[case_id]
        Y_cls = self.slide_data['cls_label'][idx]
        case_gene = torch.from_numpy(self.patient_gene[case_id + '-01'])    
        data_dir = self.data_dir

        path_features = []
        for slide_id in slide_ids:
            wsi_path  = os.path.join(data_dir, 'pt_files', '{}'.format(slide_id))   

            wsi_bag = torch.load(wsi_path)
            path_features.append(wsi_bag)

        if len(path_features) != 0:
            path_features = torch.cat(path_features, dim=0)
        else:
            for slide_id in slide_ids:
                wsi_path  = os.path.join('/data3/gzy/ganzy/data_gzy/NSCLC/', 'pt_files', '{}'.format(slide_id))   # 临时改变
                wsi_bag = torch.load(wsi_path)
                print(wsi_path)
                path_features.append(wsi_bag)
            path_features = torch.cat(path_features, dim=0)
        

        selected_elements0 = torch.from_numpy(np.array([self.patient_gene[case_id + '-01'][i] for i in self.indices0]))
        selected_elements1 = torch.from_numpy(np.array([self.patient_gene[case_id + '-01'][i] for i in self.indices1]))
        selected_elements2 = torch.from_numpy(np.array([self.patient_gene[case_id + '-01'][i] for i in self.indices2]))
        selected_elements3 = torch.from_numpy(np.array([self.patient_gene[case_id + '-01'][i] for i in self.indices3]))
        selected_elements4 = torch.from_numpy(np.array([self.patient_gene[case_id + '-01'][i] for i in self.indices4]))
        selected_elements5 = torch.from_numpy(np.array([self.patient_gene[case_id + '-01'][i] for i in self.indices5]))

        genes = []
        genes.append(selected_elements0)
        genes.append(selected_elements1)
        genes.append(selected_elements2)
        genes.append(selected_elements3)
        genes.append(selected_elements4)
        genes.append(selected_elements5)
        return (path_features, case_gene, Y_cls, Y_surv, event_time, c, case_id, genes)
    
class Generic_Split(MTDataset):
    def __init__(self, slide_data, data_dir=None, label_col=None, patient_dict=None, num_classes=2, patient_gene=None):
        self.use_h5 = False
        self.slide_data = slide_data
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.label_col = label_col
        self.patient_dict = patient_dict
        self.patient_gene = patient_gene
        self.surv_discclass_num = len(self.slide_data['label'].value_counts())
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['cls_label'] == i)[0]
        self.slide_surv_ids = [[] for i in range(self.surv_discclass_num)]
        for i in range(self.surv_discclass_num):
            self.slide_surv_ids[i] = np.where(self.slide_data['label'] == i)[0]
    
        signatures = pd.read_csv('gene_data/signatures.csv')
        Tumor_Suppressor_Genes = signatures['Tumor Suppressor Genes'].dropna().to_list()
        Oncogenes = signatures['Oncogenes'].dropna().to_list()
        Protein_Kinases = signatures['Protein Kinases'].dropna().to_list()
        Cell_Differentiation_Markers = signatures['Cell Differentiation Markers'].dropna().to_list()
        Transcription_Factors = signatures['Transcription Factors'].dropna().to_list()
        Cytokines_and_Growth_Factors = signatures['Cytokines and Growth Factors'].dropna().to_list()

        self.indices0 = [index for index, value in enumerate(self.patient_gene['gene']) if value in Tumor_Suppressor_Genes]
        self.indices1 = [index for index, value in enumerate(self.patient_gene['gene']) if value in Oncogenes]
        self.indices2 = [index for index, value in enumerate(self.patient_gene['gene']) if value in Protein_Kinases]
        self.indices3 = [index for index, value in enumerate(self.patient_gene['gene']) if value in Cell_Differentiation_Markers]
        self.indices4 = [index for index, value in enumerate(self.patient_gene['gene']) if value in Transcription_Factors]
        self.indices5 = [index for index, value in enumerate(self.patient_gene['gene']) if value in Cytokines_and_Growth_Factors]

    def __len__(self):
        return len(self.slide_data)