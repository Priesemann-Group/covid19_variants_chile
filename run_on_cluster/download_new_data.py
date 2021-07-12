# ------------------------------------------------------------------------------ #
# @Author:        Sebastian B. Mohr
# @Email:
# @Created:       2021-03-11 15:55:15
# @Last Modified: 2021-03-11 16:34:59
# ------------------------------------------------------------------------------ #
import sys
sys.path.append("../covid19_inference")
import covid19_inference as cov19

cov19.data_retrieval.retrieval.set_data_dir(fname="../data/data_covid19_inference")
jhu = cov19.data_retrieval.JHU()
jhu.download_all_available_data(force_download=True)
