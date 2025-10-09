from math import pi
import tensorflow as tf
from tensorflow import keras
from keras import backend
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, Input, Lambda, Identity

from .Kernel_Weights_Constraints import GreaterThanZeroConstraint

# =====================================================================
#
# ---------- Class implementing custom variance dense layer -----------
#
# =====================================================================


class Dense_Variance(Layer):
    """
    This class ensures that the variance converges to 2*D and shares the same weights as the Dense layer that computes the mean (see Section 5.1.1, paragraph Implementation tricks).
    """

    def __init__(self, original_layer, **kwargs):
        """
        Instantiation of a Dense_Variance Layer. This class inherits from keras Layer class.

        Arguments:
            orginial_layer: Dense layer that computes the mean
            **kwargs: standard Layer keyword arguments
        """
        super().__init__(**kwargs)

        self.original_layer = original_layer

    def call(self, inputs):
        """
        Arguments:
            inputs: multivariate traces T

        Returns:
            The variance of the monovariate traces T̃ which is computed by using
            the same weights as the Dense layer that computes the mean.

        """
        return tf.matmul(inputs, 2 * self.original_layer.weights[0])

    def get_config(self):
        """
        This methods enables a proper saving of the model.

        Returns:
            The configuration in which the arguments are serialized.

        """
        base_config = super().get_config()
        config = {
            "original_layer": keras.saving.serialize_keras_object(self.original_layer),
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        """
        This methods enables a proper saving of the model.

        Returns:
            The deserialized arguments.

        """
        original_layer_config = config.pop("original_layer")
        original_layer = keras.saving.deserialize_keras_object(original_layer_config)
        return cls(original_layer, **config)


# =====================================================================
#
# ----- Class implementing the sampling (reparametrization trick) -----
#
# =====================================================================


class Sampling(Layer):
    """
    This class corresponds to the reparametrization trick (see Section 3.3).
    It uses (z_mean, z_var) to sample the latent variable z i.e. the optimal dimensionality reduction of
    the input trace T.
    This class is inspired from https://keras.io/examples/generative/vae/.
    """

    def call(self, inputs, batch, dim):
        """
        Arguments:
            inputs: ([z_mean, z_var]) mean and variance of the monovariate trace T̃ (ie
                    optimal reduction dimensionality of T)
            dim: dimension of the input trace T

        Returns:
            A set of dim samples z which follow the multivariate Gaussian distribution N(mu_phi, Sigma_phi).
        """
        z_mean, z_var = inputs
        #batch = tf.shape(z_mean)[0]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.math.sqrt(z_var) * epsilon

    def get_config(self):
        """
        This methods enables a proper saving of the model.

        Returns:
            The configuration.

        """
        config = super().get_config()

        return config


# =====================================================================
#
# --------------- Class implementing NPcVAE-OSM model -----------------
#
# =====================================================================


class NPcVAE_OSM(Model):
    """
    This class implements a NPcVAE-OSM model ().
    It is inspired from https://keras.io/examples/generative/vae/.
    """

    def __init__(self, encoders, decoders, optimizers, nb_samples, **kwargs):
        """
        Instantiation of a NPcVAE-OSM. This class inherits from keras Model class.

        Arguments:
            encoder: encoder of a NPcVAE-OSM
            decoder: decoder of a NPcVAE-OSM
            **kwargs: standard Model keyword arguments
        """

        super(NPcVAE_OSM, self).__init__(**kwargs)

        self.encoders = encoders
        self.decoders = decoders
        self.optimizers = optimizers
        self.nb_samples = nb_samples
        self.nb_key_hypotheses = len(encoders)

        # Initialization of the learning metrics
        self.total_loss_tracker = keras.metrics.Mean(name="ELBO_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        """
        Returns:
            The learning metrics.
        """
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    @tf.function
    def train_step(self, data):
        """
        Computation of the Elbo loss terms during the training process.

        Arguments:
            data: ([input traces, orthonormal basis]) input traces T and the orthonormal bases
                   encoding their associated sensitive variable Y

        Returns:
            The result of each loss, namely the ELBO, reconstruction and KL-divergence
            losses s.t. ELBO = reconstruction + KL-divergence losses.
        """
        with tf.GradientTape(persistent=True) as tape:
            # 1) Computation of preliminary quantities that are needed for the ELBO loss computation

            ## 1.0) Getting back the dimension of input traces and the batch_size
            input_tensor = data[0]  
            dim_size = tf.shape(input_tensor)[1]
            dim = tf.cast(dim_size, dtype=tf.float32)
            batch_size = tf.shape(input_tensor)[0]

            ## 1.1) Computation of the mean/variance of the latent space Z and sampling of z ∈ Z done by the encoder
            z_means, z_vars = [], []
            for k in range(self.nb_key_hypotheses):
                z_mean, z_var = self.encoders[k]([data[0], data[1][:,k,:]])
                z_means.append(z_mean)
                z_vars.append(z_var)

            sum_z_means = tf.math.add_n(z_means)
            sum_z_vars = tf.math.add_n(z_vars)
            z_sample = Sampling()([sum_z_means, sum_z_vars], batch_size, dim_size)

            ## 1.2) Computation of the KL-divergence loss which is common for all key hypotheses
            kl_loss = (tf.math.log(2 * self.nb_key_hypotheses * dim / sum_z_vars) + \
                 ((tf.math.square(sum_z_means - self.nb_key_hypotheses * dim) + sum_z_vars) / (2 * self.nb_key_hypotheses * dim)) - 1)
            kl_loss = 0.5 * kl_loss
            kl_loss = tf.reduce_mean(kl_loss)

            total_losses, reconstruction_losses, kl_losses = [], [], []
            
            # 2) Computation for each key hypothesis of the reconstruction loss
            for k in range(self.nb_key_hypotheses):

                ## 2.1) Getting back from the encoder the weights i.e. the inverse of the variance for each sample of traces 
                encoder_weights_inverse_variance_vector = self.encoders[k].get_layer(f"z_mean").weights[0]
                encoder_weights_inverse_variance = tf.tile(encoder_weights_inverse_variance_vector, [batch_size, 1])
                encoder_weights_inverse_variance = tf.reshape(encoder_weights_inverse_variance, [batch_size, dim_size])
                encoder_weights_variance = tf.math.pow(encoder_weights_inverse_variance, -1)

                ## 2.2) Reconstruction of traces done by the decoder 
                reconstruction, decoder_psi_layer = self.decoders[k](
                    [z_sample, data[1][:,k,:], encoder_weights_variance]
                )

                ## 2.3) Computation of the reconstruction loss
                reconstruction_loss = (
                    tf.math.log(2.0 * tf.constant(pi, dtype=tf.float32) * encoder_weights_variance)
                    + tf.math.square(data[0] - decoder_psi_layer)
                    * encoder_weights_inverse_variance
                )
                reconstruction_loss = 0.5 * reconstruction_loss
                reconstruction_loss = tf.reduce_mean(
                    tf.reduce_sum(reconstruction_loss, axis=1)
                )
                
                # 3) Computation of the ELBO loss
                total_loss = reconstruction_loss + kl_loss 
                
                total_losses.append(total_loss)
                reconstruction_losses.append(reconstruction_loss)
                kl_losses.append(kl_loss)

        
        # 4) Computation and application of the gradient descent algorithm for all key hypotheses
        for k in range(self.nb_key_hypotheses):
            grad_encoder = tape.gradient(total_losses[k], self.encoders[k].trainable_weights)
            grad_decoder = tape.gradient(total_losses[k], self.decoders[k].trainable_weights)
            grads = grad_encoder + grad_decoder 
            weights = self.encoders[k].trainable_weights + self.decoders[k].trainable_weights 
            self.optimizers[k].apply_gradients(zip(grads, weights))

            
        # 5) Update of the losses values
        self.total_loss_tracker.update_state(total_losses)
        self.reconstruction_loss_tracker.update_state(reconstruction_losses)
        self.kl_loss_tracker.update_state(kl_losses)

        return {
            "ELBO loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


# =====================================================================
#
# -------------- Functions defining encoder and decoder ---------------
#
# =====================================================================

def define_encoder_variances(input_size, len_basis=256, is_deterministic=True, seed=42):
    """
    Construction of the encoder that estimates variance traces.

    Arguments:
        input_size: dimension of the input traces T
        len_basis: size of the orthonormal basis i.e. the monomial subspace (maximal degree of bit interactions=0 => len_basis=1 /
                maximal degree of bit interactions=1 => len_basis=9 / maximal degree of bit interactions=2 => len_basis=37 /
                maximal degree of bit interactions=3 => len_basis=93 / maximal degree of bit interactions=4 => len_basis=163 /
                maximal degree of bit interactions=5 => len_basis=219 / maximal degree of bit interactions=6 => len_basis=247 /
                maximal degree of bit interactions=7 => len_basis=255 / maximal degree of bit interactions=8 => len_basis=256)
        is_deterministic: boolean used to produce reproductible results. If is_deterministic is set to True, a value of seed must be specified
        seed: value of seed for reproductible results

    Returns:
        Encoder model.
    """
    if is_deterministic:
        weights_init = tf.keras.initializers.GlorotUniform(seed=seed)
    else:
        weights_init = tf.keras.initializers.GlorotUniform()

    # Input initialization (input traces, orthonormal basis)
    input_shape1 = (input_size,)
    tr_input = Input(shape=input_shape1, name="trace")

    input_shape2 = (len_basis,)
    base_input = Input(shape=input_shape2, name="orthonormal_basis_encoder")

    # Extraction of the leakage model part (psi layer)
    psi_layer = Dense(
        input_size,
        activation=None,
        use_bias=False,
        name="psi_layer_encoder",
        kernel_initializer=weights_init,
    )(base_input)

    # Noise extraction i.e. computation of the numerator of the Optimal dimensionality reduction (see Theorem 2)
    noise_layer = Lambda(
        lambda x: tf.math.square(x[0] - x[1]), name="noise_estimation"
    )([tr_input, psi_layer])

    # Output layers i.e. mean and variance of the optimal dimensionality reduction
    dense_mean = Dense(
        1,
        activation=None,
        use_bias=False,
        kernel_initializer=tf.keras.initializers.Ones(),
        kernel_constraint=GreaterThanZeroConstraint(),
        name="z_mean",
    )
    dense_mean.trainable = True

    z_mean = dense_mean(noise_layer)

    z_var = Dense_Variance(dense_mean, trainable=False, name="z_var")(noise_layer)

    # Creation of the model
    encoder = Model([tr_input, base_input], [z_mean, z_var], name="encoder")

    # encoder.summary()

    return encoder, dense_mean


def define_encoder(input_size, dense_mean, len_basis=256, is_deterministic=True, seed=42):
    """
    Construction of others encoders.

    Arguments:
        input_size: dimension of the input traces T
        dense_mean: dense_mean computed by the encoder generated using define_encoder_variances function 
        len_basis: size of the orthonormal basis i.e. the monomial subspace (maximal degree of bit interactions=0 => len_basis=1 /
                maximal degree of bit interactions=1 => len_basis=9 / maximal degree of bit interactions=2 => len_basis=37 /
                maximal degree of bit interactions=3 => len_basis=93 / maximal degree of bit interactions=4 => len_basis=163 /
                maximal degree of bit interactions=5 => len_basis=219 / maximal degree of bit interactions=6 => len_basis=247 /
                maximal degree of bit interactions=7 => len_basis=255 / maximal degree of bit interactions=8 => len_basis=256)
        is_deterministic: boolean used to produce reproductible results. If is_deterministic is set to True, a value of seed must be specified
        seed: value of seed for reproductible results

    Returns:
        Encoder model.
    """
    if is_deterministic:
        weights_init = tf.keras.initializers.GlorotUniform(seed=seed)
    else:
        weights_init = tf.keras.initializers.GlorotUniform()

    # Input initialization (input traces, orthonormal basis)
    input_shape1 = (input_size,)
    tr_input = Input(shape=input_shape1, name="trace")

    input_shape2 = (len_basis,)
    base_input = Input(shape=input_shape2, name="orthonormal_basis_encoder")

    # Extraction of the leakage model part (psi layer)
    psi_layer = Dense(
        input_size,
        activation=None,
        use_bias=False,
        name="psi_layer_encoder",
        kernel_initializer=weights_init,
    )(base_input)

    # Noise extraction i.e. computation of the numerator of the Optimal dimensionality reduction (see Theorem 2)
    noise_layer = Lambda(
        lambda x: tf.math.square(x[0] - x[1]), name="noise_estimation"
    )([tr_input, psi_layer])

    # Output layers i.e. mean and variance of the optimal dimensionality reduction
    dense_mean.trainable = False

    z_mean = dense_mean(noise_layer)

    z_var = Dense_Variance(dense_mean, trainable=False, name="z_var")(noise_layer)

    # Creation of the model
    encoder = Model([tr_input, base_input], [z_mean, z_var], name="encoder")

    # encoder.summary()

    return encoder


def define_decoder(input_size, nb_key_hypotheses=256, len_basis=256, is_deterministic=True, seed=42):
    """
    Construction of the decoder.

    Arguments:
        input_size: dimension of the input traces T
        len_basis: size of the orthonormal basis i.e. the monomial subspace (maximal degree of bit interactions=0 => len_basis=1 /
                maximal degree of bit interactions=1 => len_basis=9 / maximal degree of bit interactions=2 => len_basis=37 /
                maximal degree of bit interactions=3 => len_basis=93 / maximal degree of bit interactions=4 => len_basis=163 /
                maximal degree of bit interactions=5 => len_basis=219 / maximal degree of bit interactions=6 => len_basis=247 /
                maximal degree of bit interactions=7 => len_basis=255 / maximal degree of bit interactions=8 => len_basis=256)
        is_deterministic: boolean used to produce reproductible results. If is_deterministic is set to True, a value of seed must be specified
        seed: value of seed for reproductible results

    Returns:
        Decoder model.
    """
    if is_deterministic:
        weights_init = tf.keras.initializers.GlorotUniform(seed=seed)
    else:
        weights_init = tf.keras.initializers.GlorotUniform()

    # Input initialization (input latent representation, orthonormal basis, weights of encoder)
    input_shape1 = (input_size,)
    latent_inputs = Input(shape=input_shape1, name="z")  # z: Latent representation

    input_shape2 = (len_basis,)
    base_input = Input(shape=input_shape2, name="orthonormal_basis")

    encoder_variance = Input(shape=input_shape1, name="encoder_variance")

    # Construction of the noise step 1/2: normalization of the noise (normalized_noise_layer)
    normalized_noise_layer = Lambda(
        lambda x: (x - nb_key_hypotheses * input_size)
        / tf.math.sqrt(tf.constant(2 * nb_key_hypotheses * input_size, dtype="float32")),
        name="Normalization_noise",
    )(latent_inputs)

    # Construction of the noise step 2/2 (noise_layer)
    noise_layer = Lambda(lambda x: x[0] * tf.math.sqrt(x[1]))(
        [normalized_noise_layer, encoder_variance]
    )

    # Construction of the psi layer
    psi_layer = Dense(
        input_size,
        activation=None,
        use_bias=False,
        name="psi_layer_decoder",
        kernel_initializer=weights_init,
    )(base_input)

    # Construction of the synthetic traces (T synthetic = psi + N_theta)
    synthetic_trace = Lambda(lambda x: x[0] + x[1], name="synthetic_trace")(
        [noise_layer, psi_layer]
    )

    # Creation of the model
    decoder = Model(
        [latent_inputs, base_input, encoder_variance],
        [synthetic_trace, psi_layer],
        name="decoder",
    )

    # decoder.summary()

    return decoder

# =====================================================================
#
# --------- Function creating a NPcVAE_OSM model -----------
#
# =====================================================================
def create_model(nb_samples, nb_key_hypotheses, len_basis, learning_rate, is_deterministic=True, seeds=[42,42]):
    """
    Create a NPcVAE-OSM model.

    Arguments:
        nb_samples: number of attack traces samples
        nb_key_hypotheses: number of key hypotheses
        len_basis: number of monomials considered. Usually, len_basis is set to 2**max_nb_monomials_interactions (all monomials are considered).
                        But it can also be set to another value:
                            (maximal degree of bit interactions=0 => len_basis=1 /
                            maximal degree of bit interactions=1 => len_basis=9 / maximal degree of bit interactions=2 => len_basis=37 /
                            maximal degree of bit interactions=3 => len_basis=93 / maximal degree of bit interactions=4 => len_basis=163 /
                            maximal degree of bit interactions=5 => len_basis=219 / maximal degree of bit interactions=6 => len_basis=247 /
                            maximal degree of bit interactions=7 => len_basis=255 / maximal degree of bit interactions=8 => len_basis=256)
        learning_rate: learning rate 
        is_deterministic: boolean that allows reproductible results option. To disable reproductible results option, seeds must be set to False
        seeds: array that contains values of encoder and decoder seeds for reproductible results. To disable reproductible results option, seeds must be set to None

    Returns:
        NPcVAE-OSM instanciation
    """
    # Instanciation of encoders and decoders
    first_encoder, dense_mean = define_encoder_variances(nb_samples, len_basis, is_deterministic, seeds[0])
    encoders = [first_encoder] + [define_encoder(nb_samples, dense_mean, len_basis, is_deterministic, seeds[0]) for _ in range(1,nb_key_hypotheses)]
    decoders = [define_decoder(nb_samples, nb_key_hypotheses, len_basis, is_deterministic, seeds[1]) for _ in range(nb_key_hypotheses)]

    # Instanciation of a NPcVAE-OSM model
    optimizers = [keras.optimizers.Adam(learning_rate) for _ in range(nb_key_hypotheses)]
    NPcVAE_OSM_model = NPcVAE_OSM(encoders, decoders, optimizers, nb_samples)

    # Compilation of the model
    NPcVAE_OSM_model.compile()

    print("Model created \U00002705")

    return NPcVAE_OSM_model