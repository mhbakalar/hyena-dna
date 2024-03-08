
from pathlib import Path
from pyfaidx import Fasta
import polars as pl
import pandas as pd
import torch
from random import randrange, random
import numpy as np


"""

Dataset for sampling arbitrary intervals from the human genome.

"""


# helper functions

def exists(val):
    return val is not None

def coin_flip():
    return random() > 0.5

# augmentations

string_complement_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'a': 't', 'c': 'g', 'g': 'c', 't': 'a'}

def string_reverse_complement(seq):
    rev_comp = ''
    for base in seq[::-1]:
        if base in string_complement_map:
            rev_comp += string_complement_map[base]
        # if bp not complement map, use the same bp
        else:
            rev_comp += base
    return rev_comp


class FastaInterval():
    def __init__(
        self,
        *,
        fasta_file,
        # max_length = None,
        return_seq_indices = False,
        shift_augs = None,
        rc_aug = False,
        pad_interval = False,
    ):
        fasta_file = Path(fasta_file)
        import os
        assert fasta_file.exists(), 'path to fasta file must exist'

        self.seqs = Fasta(str(fasta_file))
        self.return_seq_indices = return_seq_indices
        # self.max_length = max_length # -1 for adding sos or eos token
        self.shift_augs = shift_augs
        self.rc_aug = rc_aug
        self.pad_interval = pad_interval        

        # calc len of each chromosome in fasta file, store in dict
        self.chr_lens = {}

        for chr_name in self.seqs.keys():
            # remove tail end, might be gibberish code
            # truncate_len = int(len(self.seqs[chr_name]) * 0.9)
            # self.chr_lens[chr_name] = truncate_len
            self.chr_lens[chr_name] = len(self.seqs[chr_name])


    def __call__(self, chr_name, start, end, max_length, neg_strand = False, return_augs = False):
        """
        max_length passed from dataset, not from init
        """
        interval_length = end - start
        chromosome = self.seqs[chr_name]
        # chromosome_length = len(chromosome)
        chromosome_length = self.chr_lens[chr_name]

        if exists(self.shift_augs):
            min_shift, max_shift = self.shift_augs
            max_shift += 1

            min_shift = max(start + min_shift, 0) - start
            max_shift = min(end + max_shift, chromosome_length) - end

            rand_shift = randrange(min_shift, max_shift)
            start += rand_shift
            end += rand_shift

        left_padding = right_padding = 0

        # checks if not enough sequence to fill up the start to end
        if interval_length < max_length:
            extra_seq = max_length - interval_length

            extra_left_seq = extra_seq // 2
            extra_right_seq = extra_seq - extra_left_seq

            start -= extra_left_seq
            end += extra_right_seq

        if start < 0:
            left_padding = -start
            start = 0

        if end > chromosome_length:
            right_padding = end - chromosome_length
            end = chromosome_length

        # Added support!  need to allow shorter seqs
        if interval_length > max_length:
            end = start + max_length

        seq = str(chromosome[start:end])

        # Add support for negative strand
        if neg_strand:
            seq = string_reverse_complement(seq)

        if self.rc_aug and coin_flip():
            seq = string_reverse_complement(seq)

        if self.pad_interval:
            seq = ('.' * left_padding) + seq + ('.' * right_padding)

        return seq

class MPRADataset(torch.utils.data.Dataset):

    '''
    Loop thru bed file, retrieve (chr, start, end), query fasta file for sequence.
    
    '''

    def __init__(
        self,
        split,
        fasta_file,
        max_length,
        dataset_name="k562",
        d_output=1, # Regression task
        val_category='positive, Smith',
        dest_path=None,
        pad_max_length=None,
        tokenizer=None,
        tokenizer_name=None,
        use_padding=None,
        add_eos=False,
        rc_aug=False,
        return_augs=False,
        return_mask=False,
    ):
        
        self.max_length = max_length
        self.use_padding = use_padding
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.d_output = d_output  # needed for decoder to grab
        self.rc_aug = rc_aug
        self.return_mask = return_mask       

        bed_path = Path(dest_path) / dataset_name / f"{split}.csv"

        assert bed_path.exists(), 'path to .bed file must exist'

        # read bed file
        df_raw = pd.read_csv(str(bed_path), sep = ',')
        print("Categories: ", df_raw['category'].unique())
        if val_category != None and split == 'val':
            print("Filtering validation set on category: ", val_category)
            df_raw = df_raw[df_raw['category'] == val_category]

        # select only split df
        self.df = df_raw[['chr.hg38','start.hg38','stop.hg38','str.hg38','mean']]
        self.df.columns = ['chr','start','end','strand','mean']

        self.fasta = FastaInterval(
            fasta_file = fasta_file,
            # max_length = max_length,
            return_seq_indices = False,
            rc_aug = rc_aug,
        )

    def __len__(self):
        return len(self.df)

    def replace_value(self, x, old_value, new_value):
        return torch.where(x == old_value, new_value, x)

    def __getitem__(self, idx):
        """Returns a sequence of specified len"""
        # sample a row from df
        row = self.df.iloc[idx]
        # row = (chr, start, end, strand, mean)
        chr_name, start, end, strand, mean = (row[0], row[1], row[2], row[3], row[4])
        neg_strand = (strand == '-')
        seq = self.fasta(chr_name, start, end, max_length=self.max_length, neg_strand=neg_strand, return_augs=self.return_augs)

        seq = self.tokenizer(seq,
            add_special_tokens=True if self.add_eos else False,  # this is what controls adding eos
            padding="max_length" if self.use_padding else "do_not_pad",
            max_length=self.max_length,
            truncation=True,
        )
        
        seq_ids = torch.LongTensor(seq["input_ids"])

        # need to wrap in list
        target = torch.FloatTensor([mean])
        
        if self.return_mask:
            return seq_ids, target, {'mask': torch.BoolTensor(seq['attention_mask'])}
        else:
            return seq_ids, target
    
class MPRASeqDataset(torch.utils.data.Dataset):

    '''
    Loop thru csv file, retrieve (seq).
    
    '''

    def __init__(
        self,
        split,
        max_length,
        dataset_name,
        kfold=None,
        d_output=1, # Regression task
        val_category=None,
        dest_path=None,
        pad_max_length=None,
        tokenizer=None,
        tokenizer_name=None,
        use_padding=None,
        add_eos=False,
        rc_aug=False,
        return_augs=False,
        return_mask=False,
        return_names=False
    ):
        
        self.max_length = max_length
        self.kfold = kfold
        self.use_padding = use_padding
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.d_output = d_output  # needed for decoder to grab
        self.rc_aug = rc_aug
        self.return_mask = return_mask
        self.return_names = return_names     

        if kfold is None:
            bed_path = Path(dest_path) / dataset_name / f"{split}.csv"
        else:
            bed_path = Path(dest_path) / dataset_name / str(kfold) / f"{split}.csv"

        assert bed_path.exists(), 'path to data file must exist'

        # read bed file
        df_raw = pd.read_csv(str(bed_path), sep = ',')
        if val_category != None and split == 'val':
            df_raw = df_raw[df_raw['category'] == val_category]

        # Select columns from input data
        self.df = df_raw[['name','category','seq','mean']]

    def __len__(self):
        return len(self.df)

    def replace_value(self, x, old_value, new_value):
        return torch.where(x == old_value, new_value, x)

    def __getitem__(self, idx):
        """Returns a sequence of specified len"""
        # sample a row from df
        row = self.df.iloc[idx]
        # row = (name, category, seq, mean)
        name, seq, mean = (row['name'], row['seq'], row['mean'])
        
        seq = self.tokenizer(seq,
            add_special_tokens=True if self.add_eos else False,  # this is what controls adding eos
            padding="max_length" if self.use_padding else "do_not_pad",
            max_length=self.max_length,
            truncation=True,
        )
        
        seq_ids = torch.LongTensor(seq["input_ids"])

        # need to wrap in list
        target = torch.FloatTensor([mean])
        
        if self.return_mask:
            #return name, seq_ids, target, {'mask': torch.BoolTensor(seq['attention_mask'])}
            if self.return_names:
                return name, seq_ids, target, {'mask': torch.BoolTensor(seq['attention_mask'])}
            else:
                return seq_ids, target, {'mask': torch.BoolTensor(seq['attention_mask'])}
        else:
            #return name, seq_ids, target, {}
            if self.return_names:
                return name, seq_ids, target
            else:
                return seq_ids, target, {} 
        
class MPRASeqSSMDataset(torch.utils.data.Dataset):

    '''
    Loop thru csv file, retrieve (seq). Mutate each basepair to all possibilties.
    
    '''

    def __init__(
        self,
        split,
        max_length,
        dataset_name,
        kfold=None,
        d_output=1, # Regression task
        val_category=None,
        dest_path=None,
        pad_max_length=None,
        tokenizer=None,
        tokenizer_name=None,
        use_padding=None,
        add_eos=False,
        rc_aug=False,
        return_augs=False,
        return_mask=False,
        return_names=False
    ):
        
        self.max_length = max_length
        self.kfold = kfold
        self.use_padding = use_padding
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.d_output = d_output  # needed for decoder to grab
        self.rc_aug = rc_aug
        self.return_mask = return_mask      
        self.return_names = return_names 

        if kfold is None:
            bed_path = Path(dest_path) / dataset_name / f"{split}.csv"
        else:
            bed_path = Path(dest_path) / dataset_name / str(kfold) / f"{split}.csv"

        assert bed_path.exists(), 'path to data file must exist'

        # read bed file
        df_raw = pd.read_csv(str(bed_path), sep = ',')
        if val_category != None and split == 'val':
            df_raw = df_raw[df_raw['category'] == val_category]

        # Select columns from input data
    
        self.df = df_raw[['name','category','seq','mean']]

        # Prebuild SSM dataset
        #self.df = self.build_ssm_dataset(self.df)

    def build_ssm_dataset(self, df):
        ssm_df = pd.DataFrame(columns=['name','category','seq','mean'])
        for i, row in df.iterrows():
            seq = row['seq']
            for index, character in enumerate(seq):
                for bp in ["A","C","T","G"]:
                    new_row = row.copy()
                    new_seq = seq[:index] + bp + seq[index+1:]
                    new_row['seq'] = new_seq
                    new_row['name'] = row['name'] + ":" + seq[index] + ":" + str(index) + ":" + bp
                    ssm_df.loc[len(ssm_df)] = new_row

        return ssm_df

    def __len__(self):
        return 4*self.max_length*len(self.df)

    def replace_value(self, x, old_value, new_value):
        return torch.where(x == old_value, new_value, x)

    def __getitem__(self, idx):
        """Returns a sequence of specified len"""

        bp_idx = idx%4
        row_idx = idx%(4*self.max_length)
        seq_idx = int(idx/(4*self.max_length))

        bp = ["A","G","T","C"][bp_idx]

        # sample a row from df
        row = self.df.iloc[row_idx]
        # row = (name, category, seq, mean)
        name, seq, mean = (row['name'], row['seq'], row['mean'])
        
        # Modify seq, name for SSM scan
        seq = seq[:seq_idx] + bp + seq[seq_idx+1:]
        name = name + ":" + seq[seq_idx] + ":" + str(seq_idx) + ":" + bp
        if bp == seq[seq_idx]:
            name += ":WT"
        
        seq = self.tokenizer(seq,
            add_special_tokens=True if self.add_eos else False,  # this is what controls adding eos
            padding="max_length" if self.use_padding else "do_not_pad",
            max_length=self.max_length,
            truncation=True,
        )
        
        seq_ids = torch.LongTensor(seq["input_ids"])

        # need to wrap in list
        target = torch.FloatTensor([mean])
        
        if self.return_mask:
            #return name, seq_ids, target, {'mask': torch.BoolTensor(seq['attention_mask'])}
            if self.return_names:
                return name, seq_ids, target, {'mask': torch.BoolTensor(seq['attention_mask'])}
            else:
                return seq_ids, target, {'mask': torch.BoolTensor(seq['attention_mask'])}
        else:
            #return name, seq_ids, target, {}
            if self.return_names:
                return name, seq_ids, target
            else:
                return seq_ids, target, {} 