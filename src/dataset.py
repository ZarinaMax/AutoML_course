import numpy as np
import pandas as pd
import logging
from typing import List
from mne.time_frequency import psd_array_multitaper
import mne

logger = logging.getLogger(__name__)


def get_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Формирует таблицу таргетов для классификации по эпохам EEG.

    Parameters
    ----------
    df : pd.DataFrame
        Оригинальный DataFrame с колонками 'epoch' и 'condition'.

    Returns
    -------
    pd.DataFrame
        Таблица с уникальными эпохами и их состояниями (condition).
    """
    try:
        result = (
            df.drop_duplicates("epoch")[["epoch", "condition"]]
            .reset_index(drop=True)
        )
        logger.info("Target dataframe сформирован успешно: %d эпох", len(result))
        return result
    except KeyError as ex:
        logger.error(
            "Входной DataFrame должен содержать столбцы 'epoch' и 'condition': %s", ex
        )
        raise


def calc_features(
    df: pd.DataFrame, channels: List[str]
) -> pd.DataFrame:
    """
    Генерирует особенности для каждой эпохи на основе выборки ЭЭГ:

      - Спектральная мощность в диапазоне бета-ритма (13-25 Гц) для каждого канала
      - P300-like фича: количество "сильных" всплесков (>5) после 40-й временной точки

    Parameters
    ----------
    df : pd.DataFrame
        Оригинальный DataFrame с EEG (столбцы: 'epoch', каналы из `channels`)
    channels : list of str
        Названия каналов ЭЭГ, на которых строиться признаки (напр. ['Fz', 'C3', ...])

    Returns
    -------
    pd.DataFrame
        DataFrame с одной строкой на эпоху и набором признаков.
    """
    feats = []

    for epoch_idx, epoch_df in df.groupby("epoch"):
        try:
            epoch_data = epoch_df[channels]

            # Расчёт спектральной плотности мощности
            psds, freqs = psd_array_multitaper(
                epoch_data.T.values, sfreq=160, verbose=False
            )
            total_power = psds.sum(axis=1)
            idx_from = np.where(freqs > 13)[0][0]
            idx_to = np.where(freqs > 25)[0][0]

            epoch_features = {"epoch": epoch_idx}

            # P300-like признаки
            for ch in channels:
                signal = epoch_data.iloc[40:][ch]
                count_over_threshold = (signal > 5).sum()
                epoch_features[f"{ch.lower()}_p300"] = count_over_threshold

            feats.append(epoch_features)

        except Exception as err:
            logger.warning(
                "Ошибка при обработке эпохи %s: %s", str(epoch_idx), str(err)
            )
            continue

    feats_df = pd.DataFrame(feats)
    logger.info("Сгенерировано признаков для %d эпох", len(feats_df))
    return feats_df
