from endersgame.accounting.pnl import Pnl
from crunch.container import StreamMessage
from enderswidgets.accounting import AccountingDataVisualizer
from enderswidgets.streams import StreamPoint, Prediction
from enderswidgets.visualization import TimeSeriesVisualizer

from typing import Union, Iterable, List, Dict, TypeVar, Any, Optional, Callable
import numpy as np

T = TypeVar('T', bound=Dict[str, Any])

def process_streams(streams: Union[Iterable[Iterable[T]], Iterable[T], Iterable[float]]) -> List[List[T]]:
    def is_iterable(obj):
        return isinstance(obj, Iterable) and not isinstance(obj, (str, bytes))

    def wrap_in_message(x: float) -> Dict[str, float]:
        return {"x": float(x)}

    # Check if streams is iterable
    if not is_iterable(streams):
        raise ValueError("Input must be iterable")

    # Handle numpy arrays
    if isinstance(streams, np.ndarray):
        if streams.ndim == 1:
            return [[wrap_in_message(x) for x in streams]]
        elif streams.ndim == 2:
            return [[wrap_in_message(x) for x in row] for row in streams]
        else:
            raise ValueError("Numpy arrays with more than 2 dimensions are not supported")

    # Try to process as Iterable[Iterable[T]]
    try:
        result = []
        for item in streams:
            if is_iterable(item) and not isinstance(item, dict):
                sub_result = []
                for sub_item in item:
                    if isinstance(sub_item, (float, np.floating, int)):
                        sub_result.append(wrap_in_message(sub_item))
                    elif isinstance(sub_item, (dict, StreamMessage)):
                        sub_result.append(sub_item)
                    else:
                        raise ValueError(f"Unsupported type in nested iterable: {type(sub_item)}")
                result.append(sub_result)
            else:
                raise TypeError  # Force to except block to handle as single stream
        return result
    except TypeError:
        # Process as single stream (Iterable[T] or Iterable[float])
        single_stream = []
        for item in streams:
            if isinstance(item, (float, np.floating, int)):
                single_stream.append(wrap_in_message(item))
            elif isinstance(item, dict):
                single_stream.append(item)
            else:
                raise ValueError(f"Unsupported type in stream: {type(item)}")
        return [single_stream]

class NoOp:
    def process(self, *args, **kwargs):
        pass
    def display(self):
        pass



def replay(streams: Union[Iterable[Iterable[T]], Iterable[T], Iterable[float]],
           horizon: int,
           update_frequency: int = 50,
           with_visualization: bool = False,
           with_accounting_visualizer: bool = False) -> Optional[float]:
    """
    Replay a set of streams, visualize the results and return the total profit
    :param update_frequency:
    :param streams:
    :param horizon:
    :param with_visualization:
    :param with_accounting_visualizer:
    :return:
    """
    try:
        from __main__ import infer
    except ImportError:
        print("Please define the 'infer' function in the main module.")
        return None
    ready_streams = process_streams(streams)
    accounting = AccountingDataVisualizer(Pnl) if with_accounting_visualizer else NoOp()
    score = 0.
    for stream_id, stream in enumerate(ready_streams):
        print(f"Processing stream {stream_id}")
        pnl = Pnl()
        viz = TimeSeriesVisualizer() if with_visualization else NoOp()
        prediction_generator = infer(stream, horizon)
        next(prediction_generator)

        for idx, data_point in enumerate(stream):
            x = data_point['x']
            prediction = next(prediction_generator)
            data = StreamPoint(substream_id=str(stream_id), value=x, ndx=idx)
            pred = Prediction(value=prediction, ndx=idx+horizon, horizon=horizon)
            accounting.process(data, pred)
            viz.process(data, pred)
            pnl.tick(data_point['x'], horizon, prediction)
            if idx % update_frequency == 0:
                accounting.display()
                viz.display()
        viz.display()
        accounting.display()
        print("Profit", stream_id, pnl.summary()['total_profit'])
        score += pnl.summary()['total_profit']
    return score
