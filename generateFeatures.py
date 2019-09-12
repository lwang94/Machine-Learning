import pandas as pd
import pickle

def generate_features(dataLocation, dataSetType, saveResults = True, saveFeatures = True):
    """Generate new features from the initial dataset. Return a fully processed dataset needed to perform machine learning on."""

    #(PMP) find total number of atoms for each molecule in the external structures dataset
    print ('Calculating number of atoms in each molecule')
    structureDataLocation = "structures.csv"
    structuredata = total_num_atoms(structureDataLocation)

    #(PMP) calculate distance between coupled atoms
    print ('Calculating atom distances')
    procData = calcBondLengths(dataLocation, structuredata)

    #(PMP) calculate distance to nearest atom
    print ('Calculating distance to nearest atoms')
    procData = minBondDist(procData)

    #generate new features through brute force engineering on different groups
    print ('Generating brute force features')
    procData, bruteforce_features0=groupby_bruteforce(procData, 'molecule_name') #(PMP) brute force feature engineering by grouping by molecule
    procData, bruteforce_features1=groupby_bruteforce(procData, 'total_num_atoms') #(PMP) brute force feature engineering by grouping by number  of atoms in molecule
    procData, bruteforce_features2=groupby_bruteforce(procData, 'min_bond_dist_binned_x') #(PMP) brute force feature engineering by grouping by distance to nearest atom
    bruteforce_features=bruteforce_features0 + bruteforce_features1 + bruteforce_features2 #keeps track of feature names

    #keep track of relevent features for machine learning in list
    relevant_features = bruteforce_features + ['bond_dist'] #(PMP) relevant features are the brute force engineering features and bond distance

    #saves processed data into .csv file
    if saveResults and dataSetType == 'train':
        print("Saving train results to CSV file.")
        procData.to_csv("trainDataPrepared.csv")
        print("Saved.")
    elif saveResults and dataSetType == 'test':
        print("Saving test results to CSV file.")
        procData.to_csv("testDataPrepared.csv")
        print("Saved.")

    #saves list of features as pickle file
    if saveFeatures:
        with open('features.pkl', 'wb') as f:
            pickle.dump(relevant_features, f)

    return procData, relevant_features

def groupby_bruteforce(DataProc, group):
    """brute force engineers new features by grouping data into specified group and calculating statistical features such as min, mean, max, etc."""
    bruteforce_features=[]

    #group data, calculate mean of bond distance of coupled atoms in each group, then put results into new column
    DataProc[f'{group}_atom_index_0_dist_mean'] = DataProc.groupby([group, 'atom_index_0'])[
        'bond_dist'].transform('mean')
    bruteforce_features+=[f'{group}_atom_index_0_dist_mean'] #keep track of name of new feature

    return DataProc, bruteforce_features

def total_num_atoms(structureDataLocation):
    """(PMP) calculate number of atoms in each molecule and adds that as feature to structure data"""
    structuredata=pd.read_csv(structureDataLocation, header=0)
    structuredata['total_num_atoms']=structuredata.groupby(['molecule_name'])['atom'].transform('count')

    return structuredata

def calcBondLengths(dataLocation, structuredata):
    """(PMP) calculate distance between coupled atoms"""

    # Load in data.
    dataRaw = pd.read_csv(dataLocation, header=0)

    # Map atomic coordinates to dataset.
    renamecolumn0 = {'atom': 'atom_0',
                            'x': 'x_0',
                            'y': 'y_0',
                            'z': 'z_0'}
    procData = map_atom_info(dataRaw, structuredata, 0, renamecolumns = renamecolumn0)

    renamecolumn1 = {'atom': 'atom_1',
                            'x': 'x_1',
                            'y': 'y_1',
                            'z': 'z_1'}
    procData = map_atom_info(procData, structuredata, 1, renamecolumns = renamecolumn1)

    procData = procData.drop('total_num_atoms_x', axis=1).rename(columns={'total_num_atoms_y':'total_num_atoms'})

    # Calculate bond length
    procData['bond_dist'] = procData.apply(lambda x: dist(x), axis=1)

    return procData

def dist(row):
    """(PMP) calculate distance"""
    return ( (row['x_1'] - row['x_0'])**2 +
             (row['y_1'] - row['y_0'])**2 +
             (row['z_1'] - row['z_0'])**2 ) ** 0.5

def minBondDist(DataProc):
    """(PMP) for each atom in the molecule, calculates the distance to the nearest other atom"""

    # create one large dataframe containing the distance between every combination of atoms in one column
    DataProc0 = pd.DataFrame(DataProc[['molecule_name', 'atom_index_0', 'atom_1', 'bond_dist']]).rename(
        columns={'atom_index_0': 'atom_index',
                 'atom_1': 'atom'})
    DataProc1 = pd.DataFrame(DataProc[['molecule_name', 'atom_index_1', 'atom_0', 'bond_dist']]).rename(
        columns={'atom_index_1': 'atom_index',
                 'atom_0': 'atom'})
    DataProc_concat = pd.concat([DataProc0, DataProc1])

    # create new columns for minimum distance and bin the min. dist. for grouping purposes later
    minDistances = DataProc_concat.sort_values('bond_dist').groupby(['molecule_name', 'atom_index'],
                                                                    as_index=False).first().rename(
        columns={'bond_dist': 'min_bond_dist'})
    minDistances['min_bond_dist_binned'] = pd.cut(minDistances['min_bond_dist'],
                                                  20)  # bin the values for grouping purposes

    # map new column onto traindata
    renamecolumn0 = {'min_bond_dist': 'min_dist_0',
                     'atom': 'min_attached_0'}
    DataProc = map_atom_info(DataProc, minDistances, 0, renamecolumns = renamecolumn0)
    renamecolumn1 = {'min_bond_dist': 'min_dist_1',
                     'atom': 'min_attached_1'}
    DataProc = map_atom_info(DataProc, minDistances, 1, renamecolumns = renamecolumn1)

    return DataProc

def map_atom_info(df, structureData, atom_idx, renamecolumns):
    """(PMP) Map atomic coordinates to dataset"""
    df = pd.merge(df, structureData, how='left',
                  left_on=['molecule_name', f'atom_index_{atom_idx}'],
                  right_on=['molecule_name', 'atom_index'])

    df = df.drop('atom_index', axis=1)
    df = df.rename(columns=renamecolumns)

    return df
