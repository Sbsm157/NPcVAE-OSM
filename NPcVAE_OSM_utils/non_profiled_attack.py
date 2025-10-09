import numpy as np
import tensorflow as tf
from .NPcVAE_OSM_tools import orthonormal_basis_projection, save_cVAE
from .NPcVAE_OSM_model import create_model
from sklearn.metrics import r2_score
from tqdm import tqdm
import gc 

# =====================================================================
#
# -------------------- Distinguishers computation ---------------------
#
# =====================================================================

def compute_distinguisher(distinguisher, coefficients, len_basis, traces=None, projection_targeted_variables=None):
    """
    Computation of distinguishers.

    Arguments:
        distinguisher: chosen distinguisher to perform LRA based attacks
            - "R2_score": R2 distinguisher
            - "R2_normalized_score": Normalized R2 distinguisher [LPR13, Section 3.2]
            - "maximum_distinguisher": Maximum distinguisher 
            - "absolute_value_sum_distinguisher": Absolute value of the sum distinguisher distinguisher 
        coefficients: estimated deterministic part coefficients. Shape expected = (number of key hypotheses * length of the basis * number of attack traces samples)
        len_basis: number of monomials considered. Usually, len_basis is set to 2**max_nb_monomials_interactions (all monomials are considered) for a generic attack.
                        But it can also be set to another value:
                            (maximal degree of bit interactions=0 => len_basis=1 /
                            maximal degree of bit interactions=1 => len_basis=9 / maximal degree of bit interactions=2 => len_basis=37 /
                            maximal degree of bit interactions=3 => len_basis=93 / maximal degree of bit interactions=4 => len_basis=163 /
                            maximal degree of bit interactions=5 => len_basis=219 / maximal degree of bit interactions=6 => len_basis=247 /
                            maximal degree of bit interactions=7 => len_basis=255 / maximal degree of bit interactions=8 => len_basis=256)
        traces: set of attack traces if chosen distinguishers are R2 or normalized R2, otherwise, this value is set to None by default
        projection_targeted_variables: projection into Walsh Hadamard basis of all hypothetical targeted variables 
                                       if chosen distinguishers are R2 or normalized R2, otherwise, this value is set to None by default
                                       Shape expected = (number of attack traces samples * number of key hypotheses * length of the basis)

    [LPR13]  Lomné, V., Prouff, E., Roche, T.: Behind the scene of side channel attacks. 
             In: Sako, K., Sarkar, P. (eds.) ASIACRYPT 2013, Part I. LNCS, vol. 8269, pp. 506–525. Springer, Heidelberg (2013)

    Returns:
        Keys ranking according to the selected distinguisher
    """   

    # Computation of distinguishers   
    match distinguisher:

        # Computation of R2 scores
        case "R2_score":
            R2_scores = np.zeros((coefficients.shape[0], traces.shape[1]))
            for k in range(coefficients.shape[0]):
                res = projection_targeted_variables[:,k,:len_basis] @ coefficients[k,:len_basis,:]
                R2_scores[k] = r2_score(traces, res, multioutput='raw_values')

            keys_ranking = np.argsort(np.max(R2_scores, axis=1))
        
        # Computation of normalized R2 scores (see [LPR13, Section 3.2])
        case "R2_normalized_score":
            R2_scores = np.zeros(coefficients.shape[0])
            for k in range(coefficients.shape[0]):
                res = projection_targeted_variables[:,k,:len_basis] @ coefficients[k,:len_basis,:]
                R2_scores[k] = r2_score(traces, res, multioutput='raw_values')

            mu_R2_scores = np.mean(R2_scores, axis=0)
            sigma_R2_scores = np.mean((R2_scores - mu_R2_scores)**2, axis=0)
            max_u = np.max(R2_scores, axis=0)
            argmax_u = np.argsort(R2_scores, axis=0)
            normalized_max = (max_u - mu_R2_scores)/sigma_R2_scores

            keys_ranking = argmax_u[:, np.argmax(normalized_max)]
        
        # Computation of maximum distinguisher
        case "maximum_distinguisher":
            keys_ranking = np.argsort(np.max(np.abs(coefficients[:,1:len_basis,:]), axis=(1,2)))
        
        # Computation of absolute value of the sum distinguisher distinguisher
        case "absolute_value_sum_distinguisher":
            keys_ranking = np.argsort(np.max(np.abs(np.sum(coefficients[:,1:len_basis,:], axis=1)),axis=1))
        
        case _:
            raise Exception("Undefined LRA distinguisher")
    
    return keys_ranking[::-1]


# =====================================================================
#
# ----------------------- Attack (Part 1/2) ---------------------------
# ---------- Estimation of deterministic parts and variance -----------
# --------------------- for all key hypotheses ------------------------
#
# =====================================================================


def run_averaged_attacks(total_nb_attack, nb_traces, attack_traces, attack_plaintexts, attack_masks, sbox, len_basis, deg_monomials_interactions, nb_key_hypotheses, path, dataset_name, \
                         learning_rate, epochs, batch_size, keys=None, is_deterministic=True, seeds=[42,42]):
    """
     Runs *total_nb_attack* attacks i.e. training of cVAE_OSM non profiled model.

    Arguments:
        total_nb_attack: total number of attacks to carry out
        nb_traces: array that contains numbers of traces on which we conduct attacks
        attack_traces: set of attack traces
        attack_plaintexts: set of attack plaintexts
        attack_masks: set of attack masks
        sbox: Sbox function
        len_basis: number of monomials considered. Usually, len_basis is set to 2**max_nb_monomials_interactions (all monomials are considered) for a generic attack.
                        But it can also be set to another value:
                            (maximal degree of bit interactions=0 => len_basis=1 /
                            maximal degree of bit interactions=1 => len_basis=9 / maximal degree of bit interactions=2 => len_basis=37 /
                            maximal degree of bit interactions=3 => len_basis=93 / maximal degree of bit interactions=4 => len_basis=163 /
                            maximal degree of bit interactions=5 => len_basis=219 / maximal degree of bit interactions=6 => len_basis=247 /
                            maximal degree of bit interactions=7 => len_basis=255 / maximal degree of bit interactions=8 => len_basis=256)
        deg_monomials_interactions: maximal degree of bit interactions. If targeted values are bytes, max_nb_monomials_interactions=8
        nb_key_hypotheses: number of key hypotheses
        path: path of the npz filename that contains all estimated deterministic parts 
        dataset_name: name of the dataset
        learning_rate: learning rate 
        epochs: number of epochs
        batch_size: batch size
        keys: Specific subset of keys if specified. If all key hypotheses are tested, this value must be set to None
        is_deterministic: boolean that allows reproductible results option. To disable reproductible results option, seeds must be set to False
        seeds: array that contains values of encoder and decoder seeds for reproductible results. To disable reproductible results option, seeds must be set to None

    """ 

    # Initialization 
    indexs_traces = np.arange(attack_traces.shape[0])
    offset_departure_subsets_traces = attack_traces.shape[0] // total_nb_attack
    index_traces_attacks = []
    attack_traces = attack_traces.astype("float32")
    
    # Computation of projection into Walsh Hadamard basis of all hypothetical targeted variables
    projection_targeted_variables_all_key_hypotheses = []
    key_hypotheses = keys if np.all(keys != None) else [i for i in range(nb_key_hypotheses)]
    for k in key_hypotheses:
        # REMINDER: The following line must be adapted to the targeted dataset
        hypothetical_targeted_variables = sbox(attack_plaintexts, k) ^ attack_masks
        # REMINDER: The following line must be adapted to the targeted dataset
        projection_targeted_variables = orthonormal_basis_projection(hypothetical_targeted_variables, deg_monomials_interactions, len_basis)
        projection_targeted_variables_all_key_hypotheses.append(projection_targeted_variables)

    projection_targeted_variables_all_key_hypotheses = np.transpose(projection_targeted_variables_all_key_hypotheses, (1,0,2))
    projection_targeted_variables_all_key_hypotheses = projection_targeted_variables_all_key_hypotheses.astype("float32")

    print("Projections for all key hypotheses done \U00002705")

    # Setting attack trace indexes for each attack
    for current_attack in range(total_nb_attack):
            index_traces_attacks.append(np.roll(indexs_traces, offset_departure_subsets_traces * current_attack))

    # Running of the attack on all numbers of traces
    for i in tqdm(range(len(nb_traces))):
        
        # Running of the attack *total_nb_attack* times considering *nb_traces[i]* traces
        for j in range(total_nb_attack):

            # Extraction of the proper attack traces subset
            index_traces_current_attack = index_traces_attacks[j]
            current_subset_index = index_traces_current_attack[:nb_traces[i]]
            traces_attack_subset = attack_traces[current_subset_index, :]
            projection_targeted_variables_all_key_hypotheses_subset = projection_targeted_variables_all_key_hypotheses[current_subset_index, :, :]

            # Creation of a model that trains on *nb_traces[i]* traces
            NPcVAE_OSM_model = create_model(attack_traces.shape[1], nb_key_hypotheses, len_basis, learning_rate, is_deterministic, seeds)

            # Training of the model
            AUTOTUNE = tf.data.AUTOTUNE
            train_dataset = tf.data.Dataset.from_tensor_slices(tuple([traces_attack_subset, projection_targeted_variables_all_key_hypotheses_subset]))
            train_dataset = (
                    train_dataset
                    .shuffle(buffer_size=10000)
                    .batch(batch_size)
                    .cache()         
                    .prefetch(AUTOTUNE)
            )
            NPcVAE_OSM_model.fit(train_dataset, epochs=epochs)

            # Saving of model parameters
            filename = f'NPcVAE_OSM_model_{dataset_name}_dataset_{nb_traces[i]}_traces_learning_rate_{learning_rate}_epochs_{epochs}_batch_size_{batch_size}_exp_{j}'
            save_cVAE(NPcVAE_OSM_model, nb_key_hypotheses, path, filename)

            # Releasing resources
            del NPcVAE_OSM_model
            del train_dataset
            gc.collect() 

            print("Training done for exp n°"+str(j+1)+" - "+str(nb_traces[i])+" traces \U00002705")

        print("Training done for "+str(nb_traces[i])+" traces \U00002705")
        
        tf.keras.backend.clear_session()
    
    print("Training done \U00002705")
    

# =====================================================================
#
# ----------------------- Attack (Part 2/2) ---------------------------
# ---- Exploitation of estimated deterministic parts and variance -----
# --------------------- for all key hypotheses ------------------------
#
# =====================================================================    


def averaged_rank(total_nb_attack, nb_traces, true_key, path, dataset_name, learning_rate, epochs, batch_size, len_basis, attack_traces, distinguisher='maximum_distinguisher', \
                  sbox=None, attack_plaintexts=None, attack_masks=None, deg_monomials_interactions=None, nb_key_hypotheses=None, keys=None):
    """
    Compute mean rank evolution for given set of estimated deterministic parts.

    Arguments:
        total_nb_attack: total number of attacks to carry out
        nb_traces: array that contains numbers of traces on which we conduct attacks
        true_key: real key value
        path: path of the npz filename that contains all estimated deterministic parts 
        dataset_name: name of the attacked dataset
        learning_rate: learning rate used to estimate deterministic parts
        epochs: number of epochs used to estimate deterministic parts
        batch_size: batch size used to estimate deterministic parts
        len_basis: number of monomials considered. Usually, len_basis is set to 2**max_nb_monomials_interactions (all monomials are considered) for a generic attack.
                        But it can also be set to another value:
                            (maximal degree of bit interactions=0 => len_basis=1 /
                            maximal degree of bit interactions=1 => len_basis=9 / maximal degree of bit interactions=2 => len_basis=37 /
                            maximal degree of bit interactions=3 => len_basis=93 / maximal degree of bit interactions=4 => len_basis=163 /
                            maximal degree of bit interactions=5 => len_basis=219 / maximal degree of bit interactions=6 => len_basis=247 /
                            maximal degree of bit interactions=7 => len_basis=255 / maximal degree of bit interactions=8 => len_basis=256).
        attack_traces: set of attack traces
        distinguisher: chosen distinguisher among proposed distinguishers in SCA literature to perform LRA based attacks
                    - "R2_score": R2 distinguisher
                    - "R2_normalized_score": Normalized R2 distinguisher [LPR13, Section 3.2]
                    - "maximum_distinguisher": Maximum distinguisher 
                    - "absolute_value_sum_distinguisher": Absolute value of the sum distinguisher distinguisher
        sbox: Sbox function if chosen distinguishers are R2 or normalized R2, otherwise, this value is set to None by default
        attack_plaintexts: set of attack plaintexts if chosen distinguishers are R2 or normalized R2, otherwise, this value is set to None by default
        attack_masks: set of attack masks if chosen distinguishers are R2 or normalized R2, otherwise, this value is set to None by default
        deg_monomials_interactions: maximal degree of bit interactions. If chosen distinguishers are R2 or normalized R2 and
                                    targeted values are bytes, max_nb_monomials_interactions=8. By default this value is set to None
        nb_key_hypotheses: number of key hypotheses. By default, this value is set to None. If chosen distinguisher are R2 or normalized R2, this value must be specified
        keys: Specific subset of keys if specified. If all key hypotheses are tested, this value must be set to None
                
    [LPR13]  Lomné, V., Prouff, E., Roche, T.: Behind the scene of side channel attacks. 
             In: Sako, K., Sarkar, P. (eds.) ASIACRYPT 2013, Part I. LNCS, vol. 8269, pp. 506–525. Springer, Heidelberg (2013)

    Returns:
        Encoder and decoder mean rank evolutions for a given distinguisher
    """
    # Initialization of mean rank evolution arrays
    encoder_key_rank_evolution = np.zeros((len(nb_traces), total_nb_attack))
    decoder_key_rank_evolution = np.zeros((len(nb_traces), total_nb_attack))

    if distinguisher=='R2_score' or distinguisher=='R2_normalized_score':
        # Initialization 
        indexs_traces = np.arange(attack_traces.shape[0])
        offset_departure_subsets_traces = attack_traces.shape[0] // total_nb_attack
        index_traces_attacks = []
        
        # Computation of projection into Walsh Hadamard basis of all hypothetical targeted variables
        projection_targeted_variables_all_key_hypotheses = []
        key_hypotheses = keys if np.all(keys != None) else [i for i in range(nb_key_hypotheses)]
        for k in key_hypotheses:
            # REMINDER: The following line must be adapted to the targeted dataset
            hypothetical_targeted_variables = sbox(attack_plaintexts, k) ^ attack_masks
            # REMINDER: The following line must be adapted to the targeted dataset
            projection_targeted_variables = orthonormal_basis_projection(hypothetical_targeted_variables, deg_monomials_interactions, len_basis)
            projection_targeted_variables_all_key_hypotheses.append(projection_targeted_variables)

        projection_targeted_variables_all_key_hypotheses = np.transpose(projection_targeted_variables_all_key_hypotheses, (1,0,2))

        # Setting attack trace indexes for each attack
        for current_attack in range(total_nb_attack):
                index_traces_attacks.append(np.roll(indexs_traces, offset_departure_subsets_traces * current_attack))

    # Running of the attack on all numbers of traces
    for i in tqdm(range(len(nb_traces))):
        
        # Running of the attack *total_nb_attack* times considering *nb_traces[i]* traces
        for j in range(total_nb_attack):

            # Extraction of estimated deterministic parts
            filename = f'NPcVAE_OSM_model_{dataset_name}_dataset_{nb_traces[i]}_traces_learning_rate_{learning_rate}_epochs_{epochs}_batch_size_{batch_size}_exp_{j}'
            estimated_psi_layers = np.load(path+filename+".npz")
            encoder_psi_layers, decoder_psi_layers = estimated_psi_layers['arr_0'], estimated_psi_layers['arr_2']

            if encoder_psi_layers.shape[2] != attack_traces.shape[1]: # Ensuring encoder_psi_layers shape fits requirements of compute_distinguisher function
                encoder_psi_layers = np.transpose(encoder_psi_layers, (0,2,1))

            if decoder_psi_layers.shape[2] != attack_traces.shape[1]: # Ensuring decoder_psi_layers shape fits requirements of compute_distinguisher function
                decoder_psi_layers = np.transpose(decoder_psi_layers, (0,2,1))

            # Computation of key rankings for both encoder and decoder estimated deterministic parts for the current attack given a specified distinguisher
            if distinguisher=='R2_score' or distinguisher=='R2_normalized_score':
                index_traces_current_attack = index_traces_attacks[j]
                current_subset_index = index_traces_current_attack[:nb_traces[i]]
                traces_attack_subset = attack_traces[current_subset_index, :]
                projection_targeted_variables_all_key_hypotheses_subset = projection_targeted_variables_all_key_hypotheses[current_subset_index, :, :]
            else :
                traces_attack_subset = None
                projection_targeted_variables_all_key_hypotheses_subset = None

            encoder_keys_ranking = compute_distinguisher(distinguisher, encoder_psi_layers, len_basis, traces_attack_subset, projection_targeted_variables_all_key_hypotheses_subset)
            decoder_keys_ranking = compute_distinguisher(distinguisher, decoder_psi_layers, len_basis, traces_attack_subset, projection_targeted_variables_all_key_hypotheses_subset)
    
            # Handling of R2 normalized distinguisher case
            if distinguisher=='R2_normalized_score':
                # Encoder case
                try:
                    if np.all(keys != None): 
                        encoder_key_rank_evolution[i,j] = np.where(true_key == keys[encoder_keys_ranking])[0][0]
                    else:
                        encoder_key_rank_evolution[i,j] = np.where(true_key == encoder_keys_ranking)[0][0]
                except: # cases where the true key does not appear as key candidate among the *attack_traces.shape[1]* samples
                    encoder_key_rank_evolution[i,j] = nb_key_hypotheses // 2

                # Decoder case
                try:
                    if np.all(keys != None): 
                        decoder_key_rank_evolution[i,j] = np.where(true_key == keys[decoder_keys_ranking])[0][0]
                    else: 
                        decoder_key_rank_evolution[i,j] = np.where(true_key == decoder_keys_ranking)[0][0]
                except: # cases where the true key does not appear as key candidate among the *attack_traces.shape[1]* samples
                    decoder_key_rank_evolution[i,j] = nb_key_hypotheses // 2
            
            # Other distinguishers
            else:
                if np.all(keys != None):
                    encoder_key_rank_evolution[i,j] = np.where(true_key == keys[encoder_keys_ranking])[0][0]
                    decoder_key_rank_evolution[i,j] = np.where(true_key == keys[decoder_keys_ranking])[0][0]

                else:
                    encoder_key_rank_evolution[i,j] = np.where(true_key == encoder_keys_ranking)[0][0]
                    decoder_key_rank_evolution[i,j] = np.where(true_key == decoder_keys_ranking)[0][0]
        
    return np.mean(encoder_key_rank_evolution, axis=1), np.mean(decoder_key_rank_evolution, axis=1)
