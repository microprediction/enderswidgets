from typing import Iterable

from enderswidgets.accounting import AccountingDataVisualizer
from enderswidgets.streams import StreamPoint, Prediction
from enderswidgets.visualization import TimeSeriesVisualizer


class NoOp:
    def process(self, *args, **kwargs):
        pass

def replay(streams: Iterable[Iterable[dict]], horizon: int, with_accounting: bool = True, with_visualization: bool = False):
    """
    Replay a set of streams and visualize the results.
    :param streams:
    :param horizon:
    :param with_accounting:
    :param with_visualization:
    :return:
    """
    try:
        from __main__ import infer
    except ImportError:
        print("Please define the 'infer' function in the main module.")
        return
    accounting = AccountingDataVisualizer() if with_accounting else NoOp()
    for stream_id, stream in enumerate(streams):
        viz = TimeSeriesVisualizer() if with_visualization else NoOp()
        prediction_generator = infer(stream, horizon)
        next(prediction_generator)
        for idx, data_point in enumerate(stream):
            prediction = next(prediction_generator)
            data = StreamPoint(substream_id=stream_id, value=data_point['x'], ndx=idx)
            pred = Prediction(value=prediction, ndx=idx+horizon, horizon=horizon)
            accounting.process(data, pred)
            viz.process(data, pred)
