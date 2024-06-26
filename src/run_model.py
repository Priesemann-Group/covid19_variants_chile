import logging

log = logging.getLogger(__name__)
import argparse
import datetime
import sys
import pymc3 as pm
import theano.tensor as tt
import pickle
import pandas as pd
import numpy as np
from multiprocessing import cpu_count

sys.path.append("../covid19_inference")

import covid19_inference

covid19_inference.data_retrieval.retrieval.set_data_dir(
    fname="../data/data_covid19_inference"
)
from covid19_inference import Cov19Model
from covid19_inference.model import (
    lambda_t_with_sigmoids,
    uncorrelated_prior_I,
    SIR,
    week_modulation,
    student_t_likelihood,
    delay_cases,
    kernelized_spread,
    kernelized_spread_variants,
    SIR_variants,
    uncorrelated_prior_E,
)
from utils import get_cps, day_to_week_matrix


def create_model(
    likelihood="dirichlet",
    spreading_dynamics="kernelized_spread",
    factor_influx=1.0,
    variants=None,
    new_cases_obs=None,
):
    """
    Creates the variant model with different compartments

    Parameters
    ----------
    likelihood : str
        Likelihood function for the variant data, possible : ["beta","binom","dirichlet","multinomial"]

    spreading_dynamics : str
        Type of spreading dynamics to use possible : ["SIR","kernelized_spread"]

    variants : pd.DataFrame
        Data array variants

    new_cases : pd.DataFrame
        Data array cases

    Returns
    -------
    model

    """
    if variants is None:
        # Load data variants
        variants = pd.read_excel(
            "../data/Chile_Variants_Updated_with_airports.xlsx",
            sheet_name="Variants_Count",
        )
        variants = variants.set_index("Lineage").T
        variants.index.name = "Week"
        variants.index = pd.to_datetime(variants.index + "-1", format="%V_%G-%u")
        variants = variants.iloc[0:-1]

    if new_cases_obs is None:
        # Load casenumbers chile and sum over weeks
        jhu = covid19_inference.data_retrieval.JHU()
        jhu.download_all_available_data(force_local=True)
        new_cases_obs = jhu.get_new(
            country="Chile", data_begin=variants.index[0], data_end=variants.index[-1]
        )

    # DataRange
    data_begin = new_cases_obs.index[0]

    # Params for the model
    params = {
        "new_cases_obs": new_cases_obs,
        "data_begin": data_begin,
        "fcast_len": 42,
        "diff_data_sim": 16,
        "N_population": 19276715,  # population chile
    }

    # Variant data
    variant_names = ["B.1.1", "B.1.1.348", "B.1.1.7", "C.37", "P.1"]
    num_variants = 5
    dat_variants_unknown = np.array(variants["N_Total"]) - np.array(
        variants[variant_names]
    ).sum(axis=1)
    dat_variants_known = np.array(variants[variant_names])
    dat_variants = np.concatenate(
        (dat_variants_known, dat_variants_unknown[:, np.newaxis]), axis=1
    )
    dat_total = np.stack(
        [np.array(variants["N_Total"])] * (len(variant_names) + 1), axis=1
    )

    # Calculate the scaling of the influx, we use the cases of the neighbour countries
    # this function is called in the model because we need it in the range sim begin and sim end
    def get_neighbour(be, en):
        jhu = covid19_inference.data_retrieval.JHU()
        jhu.download_all_available_data(force_local=True)
        cases = jhu.get_new(
            country="Argentina", data_begin=be - datetime.timedelta(days=7), data_end=en
        )
        cases += jhu.get_new(
            country="Brazil", data_begin=be - datetime.timedelta(days=7), data_end=en
        )
        cases += jhu.get_new(
            country="Peru", data_begin=be - datetime.timedelta(days=7), data_end=en
        )
        # extend cases by the last value
        missing_days = [
            cases.index[-1] + datetime.timedelta(i)
            for i in range(1, (en - cases.index[-1]).days + 1)
        ]
        avg = cases.rolling(7).mean()
        copied_last_day = avg.iloc[[-1] * len(missing_days)]
        return avg.append(copied_last_day.set_axis(missing_days))[be:en]

    pr_delay = 10

    if spreading_dynamics == "SIR":
        pr_median_lambda = 0.125
    elif "kernelized_spread" in spreading_dynamics:
        pr_median_lambda = 1.0

    with Cov19Model(**params) as this_model:

        # Get base reproduction number/spreading rate
        lambda_t_log = lambda_t_with_sigmoids(
            change_points_list=get_cps(
                this_model.data_begin,
                this_model.sim_end,
                interval=14,
                pr_median_transient_len=6,
                pr_sigma_transient_len=2,
            ),
            pr_median_lambda_0=pr_median_lambda,
            name_lambda_t="base_lambda_t",
        )
        lambda_t_unknown_log = lambda_t_with_sigmoids(
            change_points_list=get_cps(
                this_model.data_begin,
                this_model.sim_end,
                interval=35,
                pr_median_transient_len=20,
                pr_sigma_transient_len=6,
                pr_sigma_date_transient=5,
            ),
            pr_median_lambda_0=1,
            pr_sigma_lambda_0=0.2,
            name_lambda_t="unknown_lambda_t",
            prefix_lambdas="un_",
        )

        # Factor for each linage
        f = pm.Lognormal(name="f_v", mu=0, sigma=1, shape=(len(variant_names)))
        f = f * np.array([1, 1, 0, 1, 1]) + np.array([0, 0, 1, 0, 0])

        # Influx
        week_mapping = day_to_week_matrix(
            this_model.sim_begin,
            this_model.sim_end,
            [
                this_model.sim_begin + datetime.timedelta(weeks=i)
                for i in range(-(-this_model.sim_len // 7))
            ],
        )

        variants_mapping = day_to_week_matrix(
            this_model.sim_begin,
            this_model.sim_end,
            variants.index,
        )

        if spreading_dynamics == "SIR":
            # Influx
            Phi_w = pm.HalfStudentT(
                "Phi_w",
                sigma=0.0005 * factor_influx,
                nu=4,
                shape=(num_variants, week_mapping.shape[1]),
            )
            Phi = Phi_w.dot(week_mapping.T) * get_neighbour(
                this_model.sim_begin, this_model.sim_end
            )
            Phi = pm.Deterministic("Phi", Phi.T)

            # Adds the recovery rate mu to the model as a random variable
            mu = pm.Lognormal(name="mu", mu=np.log(1 / 8), sigma=0.2)

            # This builds a decorrelated prior for I_begin for faster inference.
            prior_I = uncorrelated_prior_I(
                lambda_t_log=lambda_t_log + lambda_t_unknown_log,
                mu=mu,
                pr_median_delay=pr_delay,
            )

            # Construct SIR models, for each variant and for the unknow variants
            new_cases_unknown = SIR(
                lambda_t_log=lambda_t_log + lambda_t_unknown_log,
                mu=mu,
                name_new_I_t="new_I_t",
                pr_I_begin=prior_I,
                name_I_t="I_t",
                name_S_t="unS_t",
            )

            new_cases_v = SIR_variants(
                lambda_t_log=lambda_t_log,
                mu=mu,
                f=f,
                num_variants=num_variants,
                name_I_begin="I_begin_v",
                Phi=Phi,
            )
            # Put the new cases together unknown and known into one tensor (shape: t,v)

            new_cases_v = pm.Deterministic(
                "new_cases_v",
                tt.concatenate(
                    [
                        new_cases_v,
                        new_cases_unknown[:, None],
                    ],
                    axis=1,
                ),
            )
        elif spreading_dynamics == "kernelized_spread":
            # Influx
            Phi_w = pm.HalfStudentT(
                "Phi_w",
                sigma=0.0005 * factor_influx,
                nu=4,
                shape=(num_variants + 1, week_mapping.shape[1]),
            )
            Phi = Phi_w.dot(week_mapping.T) * get_neighbour(
                this_model.sim_begin, this_model.sim_end
            )
            Phi = pm.Deterministic("Phi", Phi.T)

            # Put the lambdas together unknown and known into one tensor (shape: t,v)
            new_cases_v = kernelized_spread_variants(
                R_t_log=tt.concatenate(
                    [
                        lambda_t_log[:, None] * np.ones(num_variants),
                        lambda_t_log[:, None] + lambda_t_unknown_log[:, None],
                    ],
                    axis=1,
                ),
                f=tt.concatenate([f, tt.as_tensor_variable([1])]),
                num_variants=num_variants + 1,
                # pr_mean_median_incubation=mean_median_incubation,
                # pr_sigma_median_incubation=None,
                Phi=Phi,
                pr_new_E_begin=250,
            )

        elif spreading_dynamics == "kernelized_spread_and_changing_factors":
            sigma_change_factors = (
                pm.HalfNormal(
                    name="sigma_change_factors",
                    sigma=1,
                    transform=pm.transforms.log_exp_m1,
                )
            ) * 0.05

            slow_change_factors = lambda_t_with_sigmoids(
                change_points_list=get_cps(
                    this_model.data_begin,
                    this_model.data_end,
                    interval=50,
                    pr_median_transient_len=30,
                    pr_sigma_transient_len=8,
                    pr_sigma_date_transient=8,
                ),
                pr_median_lambda_0=1,
                pr_sigma_lambda_0=0.001,
                name_lambda_t="change_of_factors",
                prefix_lambdas="f_",
                hierarchical=False,
                sigma_lambda_cp=sigma_change_factors,
                shape=num_variants,
            )
            f = np.exp(slow_change_factors) * f
            Phi_w = pm.HalfStudentT(
                "Phi_w",
                sigma=0.0005 * factor_influx,
                nu=4,
                shape=(num_variants + 1, week_mapping.shape[1]),
            )
            Phi = Phi_w.dot(week_mapping.T) * get_neighbour(
                this_model.sim_begin, this_model.sim_end
            )
            Phi = pm.Deterministic("Phi", Phi.T)

            # Put the lambdas together unknown and known into one tensor (shape: t,v)
            new_cases_v = kernelized_spread_variants(
                R_t_log=tt.concatenate(
                    [
                        lambda_t_log[:, None] * np.ones(num_variants),
                        lambda_t_log[:, None] + lambda_t_unknown_log[:, None],
                    ],
                    axis=1,
                ),
                f=tt.concatenate([f, tt.ones((this_model.sim_len, 1))], axis=1),
                num_variants=num_variants + 1,
                Phi=Phi,
                f_is_varying=True,
                pr_new_E_begin=250,
            )

        # Delay the cases by a lognormal reporting delay and add them as a trace variable
        new_cases_v = delay_cases(
            cases=new_cases_v,
            name_cases="delayed_cases",
            pr_mean_of_median=pr_delay,
            pr_median_of_width=0.3,
            num_variants=num_variants + 1,  # +1 beause of unknown
        )

        # Calculate total reported new cases (sum all variants)
        new_cases = new_cases_v.sum(axis=1)

        # Modulate the inferred cases by a abs(sin(x)) function, to account for weekend effects
        # Also adds the "new_cases" variable to the trace that has all model features.
        new_cases_modulated = week_modulation(cases=new_cases, name_cases="new_cases")

        # Define the likelihood, uses the new_cases_obs set as model parameter
        student_t_likelihood(cases=new_cases_modulated)

        # Calculate daily fraction of each variant tau
        tau = new_cases_v / new_cases[:, np.newaxis]
        pm.Deterministic("tau", tau)

        # Map daily tau to match weekly data
        tau_w = tau.T.dot(variants_mapping).T / 7  # mean tau value
        tau_w = tt.clip(tau_w, 1e-3, 0.9999)  # bad starting energy fix range: (0,1)
        tau_w /= tau_w.sum(axis=1, keepdims=True)

        pm.Deterministic("tau_w", tau_w)

        # Variants data
        y = pm.Data("y_obs", dat_variants)
        n = pm.Data("n_obs", dat_total)

        # Likelihood for variants
        if likelihood == "beta":
            logp = pm.Beta.dist(alpha=y + 1, beta=n - y + 1).logp(tau_w)
            error = pm.Potential("error_tau", logp)
        elif likelihood == "binom":
            pm.Binomial(
                "tau_w_obs",
                p=tau_w,
                observed=y,
                n=n,
                shape=(len(variants), len(variant_names)),
            )
        elif likelihood == "dirichlet":
            factor = pm.Lognormal(
                "factor_likelihood",
                mu=np.log(0.8),
                sigma=0.15,
                transform=pm.transforms.log_exp_m1,
            )
            logp = pm.Dirichlet.dist(a=y * factor + 1).logp(tau_w)
            error = pm.Potential("error_tau", logp)

        elif likelihood == "multinomial":
            pm.Multinomial(
                "tau_w_obs",
                p=tau_w,
                observed=y,
                n=n,
            )

        return this_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model")

    parser.add_argument(
        "-l",
        "--likelihood",
        type=str,
        help='Possible: ["beta","binom","dirichlet","multinomial"]',
        default="dirichlet",
    )

    parser.add_argument(
        "-s",
        "--spread_method",
        type=str,
        help='Possible: ["SIR","kernelized_spread"]',
        default="kernelized_spread",
    )

    parser.add_argument("--log", type=str, help="Log directory", default="../run_on_cluster/log")

    args = parser.parse_args()

    """ Basic logger setup
    We want each job to print to a different file, for easier debuging once
    run on a cluster.
    """
    f_str = "Variants_v2"
    for arg in args.__dict__:
        if arg in [
            "log",
        ]:
            continue
        f_str += f"-{arg}={args.__dict__[arg]}"

    # Write all logs to file
    fh = logging.FileHandler(args.log + "/" + f_str + ".log")
    fh.setFormatter(
        logging.Formatter("%(asctime)s::%(levelname)-4s [%(name)s] %(message)s")
    )
    fh.setLevel(logging.DEBUG)
    log.addHandler(fh)
    covid19_inference.log.addHandler(fh)
    log.info(f"Script started: {datetime.datetime.now()}")
    log.info(f"Args: {args.__dict__}")

    # Redirect all errors to file
    sys.stderr = open(args.log + "/" + f_str + ".stderr", "w")
    sys.stdout = open(args.log + "/" + f_str + ".stdout", "w")

    # Redirect pymc3 output
    logPymc3 = logging.getLogger("pymc3")
    logPymc3.addHandler(fh)

    # Create model
    model = create_model(args.likelihood, args.spread_method)

    # Sample
    multitrace, trace, multitrace_tuning, trace_tuning = covid19_inference.robust_sample(
        model,
        draws=600,
        tune=600,
        tune_2nd=700,
        tuning_chains=12,
        final_chains=6,
        return_tuning=True,
        max_treedepth=10,
        target_accept=0.95,
    )

    # Save trace/model so we dont have to rerun sampling every time we change some plotting routines
    with open(f"../data/pickled/{f_str}_forecast.pickle", "wb") as f:
        pickle.dump((model, trace), f)
