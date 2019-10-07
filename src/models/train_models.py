import sys

sys.path.insert(0, "/home/emmanuel/code/rbig")

from rbig import RBIG, RBIGMI, RBIGKLD
import numpy as np
import time


def run_rbig_models(X1, X2=None, measure="t", verbose=None, random_state=123):

    # RBIG Parameters
    n_layers = 10000
    rotation_type = "PCA"
    zero_tolerance = 60
    pdf_extension = 10
    pdf_resolution = None
    tolerance = None

    if measure.lower() == "t" or measure.lower() == "h":

        # RBIG MODEL 0
        rbig_model = RBIG(
            n_layers=n_layers,
            rotation_type=rotation_type,
            random_state=random_state,
            zero_tolerance=zero_tolerance,
            tolerance=tolerance,
            pdf_extension=pdf_extension,
            pdf_resolution=pdf_resolution,
            verbose=verbose,
        )

        # fit model to the data
        t0 = time.time()
        rbig_model.fit(X1)
        t1 = time.time() - t0
        if verbose:
            print(
                f"Trained RBIG ({X1.shape[0]:,} points, {X1.shape[1]:,} dimensions): {t1:.3f} secs"
            )

        tc = rbig_model.mutual_information * np.log(2)
        h = rbig_model.entropy(correction=True) * np.log(2)

        if verbose:
            print(f"TC: {tc:.3f}")
            print(f"H: {h:.3f}")

        return tc, h, t1

    elif measure.lower() == "mi":
        # RBIG MODEL 0
        rbig_mi_model = RBIGMI(
            n_layers=n_layers,
            rotation_type=rotation_type,
            random_state=random_state,
            zero_tolerance=zero_tolerance,
            tolerance=tolerance,
            pdf_extension=pdf_extension,
            pdf_resolution=pdf_resolution,
            verbose=verbose,
        )

        # fit model to the data
        t0 = time.time()
        rbig_mi_model.fit(X1, X2)
        t1 = time.time() - t0

        if verbose:
            print(
                f"Trained RBIG1 MI ({X1.shape[0]:,} points, {X1.shape[1]:,} dimensions): {t1:.3f} secs"
            )

        mi = rbig_mi_model.mutual_information() * np.log(2)
        if verbose:
            print(f"MI: {mi:.3f}")

        return mi, t1

    elif measure.lower() == "kld":

        rbig_kld_model = RBIGKLD(
            n_layers=n_layers,
            rotation_type=rotation_type,
            random_state=random_state,
            zero_tolerance=zero_tolerance,
            tolerance=tolerance,
            pdf_extension=pdf_extension,
            pdf_resolution=pdf_resolution,
            verbose=verbose,
        )

        # fit model to the data
        t0 = time.time()
        rbig_kld_model.fit(X1, X2)
        t1 = time.time() - t0

        if verbose:
            print(
                f"Trained RBIG KLD ({X1.shape[0]:,}, {X2.shape[0]:,} points): {t1:.3f} secs"
            )

        kld = rbig_kld_model.kld * np.log(2)
        
        if verbose:
            print(f"KLD: {kld:.3f}")

        return kld, t1

    else:
        raise ValueError(f"Unrecognized measure: {measure}.")

