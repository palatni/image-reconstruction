"""
This script monitors the cache folder, plots charts with metrics and shows
the current image, restored from the target during training. csv_metric
and prediction_image keys should be specified in the config file.
"""
from typing import Dict
import sys
from time import sleep
from pathlib import Path
import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from utils import load_image, load_yaml


class _ChartSection:
    def __init__(self, csv_path: Path) -> None:
        self._csv_path = csv_path

        st.markdown(
            body='<h2 style=\'text-align: center'
            '; color: white;\'>Training Charts</h2>',
            unsafe_allow_html=True,
        )
        self._mse_chart, self._psnr_chart = st.columns(2)
        self._mse_chart = self._mse_chart.empty()
        self._psnr_chart = self._psnr_chart.empty()
        self.update_section()

    def update_section(self) -> None:
        if self._csv_path.is_file():
            df = pd.read_csv(self._csv_path)
            mse_data = df[['Train MSE', 'Val MSE']]
            mse_data.columns = ['Train', 'Val']
            psnr_data = df[['Train PSNR', 'Val PSNR']]
            psnr_data.columns = ['Train', 'Val']
            self._mse_chart.line_chart(
                data=mse_data,
                x_label='Epoch id',
                y_label='MSE',
            )
            self._psnr_chart.line_chart(
                data=psnr_data,
                x_label='Epoch id',
                y_label='PSNR',
            )
        else:
            self._mse_chart.line_chart(x_label='Epoch id', y_label='MSE')
            self._psnr_chart.line_chart(x_label='Epoch id', y_label='PSNR')


class _ImgSection:
    def __init__(self, img_path: Path, pred_img_path: Path) -> None:
        img = load_image(img_path)
        self._pred_img_path = pred_img_path
        img_col, self._pred_img_col = st.columns(2)
        img_col.header('Target Image')
        self._pred_img_col.header('Predicted Image')
        self._pred_img_col = self._pred_img_col.empty()

        img_fig, img_ax = plt.subplots()
        img_ax.imshow(img)
        img_col.pyplot(img_fig)

        self._pred_fig, self._pred_ax = plt.subplots()

    def update_section(self) -> None:
        if self._pred_img_path.is_file():
            pred_img = load_image(self._pred_img_path)
            self._pred_ax.imshow(pred_img)
            self._pred_img_col.pyplot(self._pred_fig)


def _parse_args() -> Dict:
    """
    Unfortunately Streamlit disallows flag arguments:
    https://discuss.streamlit.io/t/command-line-arguments/386/2
    So, ArgumentParser can't be applied here

    Returns:
        Dict: Configuration dictionary.
    """
    if len(sys.argv) == 1:
        return 'config.yaml'
    file_path = Path(sys.argv[1])
    if not file_path.is_file():
        raise ValueError(f'{file_path} is not a file')
    return file_path


if __name__ == '__main__':
    config_path = _parse_args()
    config = load_yaml(config_path)

    cache_dir = Path(config['cache_dir'])
    cache_dir.mkdir(parents=True, exist_ok=True)

    img_path = config['source_img']
    csv_path = cache_dir / config['csv_metric']['file']
    pred_img_path = cache_dir / config['predicted_image']['file']
    chart_section = _ChartSection(csv_path)
    img_section = _ImgSection(img_path, pred_img_path)

    while True:
        chart_section.update_section()
        img_section.update_section()
        sleep(config['update_delay'])
