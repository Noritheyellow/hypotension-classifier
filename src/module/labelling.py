from itertools import groupby
import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view
from abc import ABC, abstractmethod


class LabelStrategy(ABC):
    def label_event(self):
        pass

    def label_pos_event(self):
        pass

    def label_neg_event(self):
        pass

    def extract_pos_segment(self):
        pass

    def extract_neg_segment(self):
        pass

    def extract_segment(self):
        pass


class Hypophet(LabelStrategy):
    name = "hypophet"

    def __init__(self):
        super().__init__()

    def set_window_of(self, arr, win_size) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Windowing input array with constant win_size.
        """
        if len(arr) < win_size:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
        windows = sliding_window_view(arr, window_shape=win_size)[::win_size]
        windows_idx = np.arange(windows.size).reshape(-1, win_size)
        # end point 에 1을 더하여 이후 배열을 생성할 때 용이하도록 한다.
        windows_range = np.hstack([windows_idx[:, 0:1], windows_idx[:, -1:] + 1])
        return windows, windows_range

    def extend_to_past(self, events, past_len) -> npt.NDArray:
        past_loc = events[:, 0:1] - past_len
        start = np.where(past_loc < 0, 0, past_loc)
        end = events[:, -1:]
        return np.hstack([start, end], dtype=np.int32)

    def group_event(self, events, config) -> npt.NDArray:
        """
        * Make range for grouping for each event.
        * Events intersects with each others get a group number.
        *
        * For example, if segment A's range is (1200, 2500) and segment B's range is (2400, 3000),
        * each segment is intersect at (2400, 2500).
        * Thus they can be merged into new segment with range (1200, 3000).
        """
        event_cond = config["event_condition"]["pos"]
        grp_rng = event_cond["group_range"]

        grp_len = config["fs"] * 60 * grp_rng
        events_with_grp_rng = [
            (event[0] - grp_len, event[-1] + grp_len) for event in events
        ]

        group_no = np.zeros(len(events_with_grp_rng))
        for i in range(len(events_with_grp_rng) - 1):
            segment = events_with_grp_rng[i]
            next_segment = events_with_grp_rng[i + 1]
            start, end = segment[0], segment[-1]

            if start <= next_segment[0] < end:
                group_no[i + 1] = group_no[i]
            else:
                group_no[i + 1] = group_no[i] + 1

        events_with_grp = tuple(zip(events, group_no))
        grp_events = []
        for _, grp in groupby(events_with_grp, lambda x: x[1]):
            temp_grp = list(grp)
            start_idx, end_idx = temp_grp[0][0][0], temp_grp[-1][0][-1]
            grp_events.append((start_idx, end_idx))
        return np.vstack(grp_events, dtype=np.int32)

    def apply_adjacency(self, idx_arr, adj, sig_len, input_len) -> npt.NDArray:
        adj_events = []
        for e in idx_arr:
            left, right = e[0], e[-1]

            if left != 0:
                left = left + adj
            if right != sig_len - 1:
                right = right - adj
            if left >= right:
                continue
            else:
                adj_events.append([left, right + 1])

        events = list(filter(lambda e: (e[-1] - e[0]) >= input_len, adj_events))
        return np.vstack(events, dtype=np.int32)

    def label_pos_event(self, arr, config) -> npt.NDArray:
        """
        Label positive event from input array based on config file.

        1. Slice array with window size.
        2. Give condition to each windows.
        3. Label positive event(step 1): avg(MAP) < 65 at least 1 minute.
        4. Label positive event(step 2): extend to past N(target predict length) minute.
        5. Label positive event(step 3): merge adjacent positive events.
        """
        fs = config["fs"]
        input_len, pred_len = config["input_len"] * fs, config["pred_len"] * fs
        win_size = config["event_condition"]["pos"]["N"] * fs
        windows, win_rng = self.set_window_of(arr, win_size)
        is_window_valid = np.all(windows != 0, axis=1)
        is_positive = windows.mean(axis=1) < 65
        try:
            pos_event = win_rng[is_window_valid & is_positive]
            pos_event = self.extend_to_past(pos_event, input_len + pred_len)
            pos_event = self.group_event(pos_event, config)
        except Exception as e:
            pos_event = np.array([], dtype=np.int32)

        return pos_event

    def label_neg_event(self, arr, pos_event, config) -> npt.NDArray:
        input_len = config["input_len"] * config["fs"]
        adj = config["event_condition"]["neg"]["adjacency"] * 60 * config["fs"]
        raw_idx = np.arange(arr.size)
        if len(pos_event) != 0:
            pos_idx = np.hstack([np.arange(*e) for e in pos_event])
        else:
            pos_idx = np.array([], dtype=np.int32)

        try:
            neg_idx = np.setdiff1d(raw_idx, pos_idx)
            split_loc = np.where(np.diff(neg_idx, prepend=neg_idx[0] - 1) != 1)[0]
            split_neg_idx = np.split(neg_idx, split_loc)
            neg_event = self.apply_adjacency(
                split_neg_idx, adj, raw_idx.size, input_len
            )  # (N, t)

        except Exception as e:
            neg_event = np.array([], dtype=np.int32)

        return neg_event

    def extract_segment(self, events, config, label):
        fs = config["fs"]
        input_len = config["input_len"] * fs
        # TODO: 만약 moghadam 처럼 input_len 간격으로 샘플링을 한다면? -> 딱히 차이 없다.
        stride = config["segment"]["seg_stride"] * fs * config["segment"]["seg_density"]
        # stride = input_len

        if len(events) == 0:
            return np.array([], dtype=np.int32)

        seg_idx = [sliding_window_view(range(*e), input_len)[::stride] for e in events]
        seg_idx = np.vstack(seg_idx)
        seg_range = np.hstack([seg_idx[:, 0:1], seg_idx[:, -1:] + 1])
        return seg_range


class HypophetVersion2(LabelStrategy):
    name = "hypophetversion2"

    def __init__(self):
        super().__init__()

    def set_window_of(self, arr, win_size) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Windowing input array with half-overlapping windows.
        """
        if len(arr) < win_size:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
        windows = sliding_window_view(arr, win_size)[:: win_size // 2]
        windows_idx = sliding_window_view(np.arange(arr.size), win_size)[
            :: win_size // 2
        ]
        # end point 에 1을 더하여 이후 배열을 생성할 때 용이하도록 한다.
        windows_range = np.hstack([windows_idx[:, 0:1], windows_idx[:, -1:] + 1])
        return windows, windows_range

    def group_event(self, events, config, grp_rng) -> npt.NDArray:
        """
        * Make range for grouping for each event.
        * Events intersects with each others get a group number.
        *
        * For example, if segment A's range is (1200, 2500) and segment B's range is (2400, 3000),
        * each segment is intersect at (2400, 2500).
        * Thus they can be merged into new segment with range (1200, 3000).
        """
        grp_len = config["fs"] * 60 * grp_rng
        events_with_grp_rng = [
            (event[0] - grp_len, event[-1] + grp_len) for event in events
        ]

        group_no = np.zeros(len(events_with_grp_rng))
        for i in range(len(events_with_grp_rng) - 1):
            segment = events_with_grp_rng[i]
            next_segment = events_with_grp_rng[i + 1]
            start, end = segment[0], segment[-1]

            if start <= next_segment[0] < end:
                group_no[i + 1] = group_no[i]
            else:
                group_no[i + 1] = group_no[i] + 1

        events_with_grp = tuple(zip(events, group_no))
        grp_events = []
        for _, grp in groupby(events_with_grp, lambda x: x[1]):
            temp_grp = list(grp)
            start_idx, end_idx = temp_grp[0][0][0], temp_grp[-1][0][-1]
            grp_events.append((start_idx, end_idx))
        return np.vstack(grp_events, dtype=np.int32)

    def mark_actual_pos_event(self, config, arr, grp_rng) -> npt.NDArray:
        """
        slide 1-minute half-overlapping window defining hypotension event
        window: t ≤ x < t + 6000
        """
        fs = config["fs"]
        win_size = config["event_condition"]["pos"]["N"] * fs

        try:
            windows, win_rng = self.set_window_of(arr, win_size)
            is_window_valid = np.all(windows != 0, axis=1)
            is_positive = windows.mean(axis=1) < 65
            actual_pos_event = win_rng[is_window_valid & is_positive]
            actual_pos_event = self.group_event(actual_pos_event, config, grp_rng)
            actual_pos_event = np.where(actual_pos_event < 0, 0, actual_pos_event)
        except Exception as e:
            print(f"Error occurred while processing positive events: {e}")
            actual_pos_event = np.array([], dtype=np.int32)

        return actual_pos_event

    def mark_gray_zone(self, config, arr, ref_event) -> tuple[npt.NDArray, npt.NDArray]:
        """
        gray zone은 actual pos event의 양끝에서 N분으로 정의된다.
        만약 gray zone이 다른 actual pos event와 겹친다면, actual pos event 전까지만 gray zone으로 정의한다.
        """
        fs = config["fs"]
        adj = fs * 60 * config["adjacency"]

        soft_gray_zone, hard_gray_zone = [], []
        for e in ref_event:
            left_end, right_start = e[0], e[1]
            left_start, right_end = left_end - adj, right_start + adj
            soft_gray_zone.append((left_start, left_end))
            hard_gray_zone.append((right_start, right_end))

        try:
            soft_gray_zone = np.vstack(soft_gray_zone)
            soft_gray_zone = np.where(soft_gray_zone < 0, 0, soft_gray_zone)
        except ValueError:
            print("No soft gray zones defined.")
            soft_gray_zone = np.array([], dtype=np.int32)

        try:
            hard_gray_zone = np.vstack(hard_gray_zone)
            hard_gray_zone = np.where(
                hard_gray_zone >= len(arr), len(arr), hard_gray_zone
            )
        except ValueError:
            print("No hard gray zones defined.")
            hard_gray_zone = np.array([], dtype=np.int32)

        return soft_gray_zone, hard_gray_zone

    def mark_predicted_pos_event(self, config, ref_event) -> npt.NDArray:
        """
        actual pos event의 시작부분으로부터 N분 lookback으로 정의된다.
        15분 lookback으로부터 30초 non-overlapping window를 추출한다고 하면,
        마지막 새그먼트 즉, actual과 겹치는 인덱스를 가지는 새그먼트는 배제되어야 한다.
        predicted window: t ≤ x < t + pred_len
        """
        fs = config["fs"]
        pred_len = config["pred_len"] * fs
        pred_pos_event = []
        for e in ref_event:
            pred_end = e[0]
            pred_start = pred_end - pred_len
            pred_pos_event.append((pred_start, pred_end))

        try:
            pred_pos_event = np.vstack(pred_pos_event)
            pred_pos_event = np.where(pred_pos_event < 0, 0, pred_pos_event)
        except ValueError:
            print("No predicted positive events defined.")
            pred_pos_event = np.array([], dtype=np.int32)

        return pred_pos_event

    def mark_on_label_map(self, arr, layers: list) -> npt.NDArray:
        """
        각 라벨의 index를 먼저 구한다.
        그것을 정해진 순서에 따라 signal에 레이어를 쌓듯 쌓아올린다.
        non-hypotension -> soft gray zone -> predicted -> hard gray zone -> actual
        actual pos(2), predicted pos(1), neg(0), soft gray zone(-1), hard gray zone(-2)
        """
        label_map = np.zeros_like(arr)

        for e in layers[0]:
            label_map[range(*e)] = -1

        for e in layers[1]:
            label_map[range(*e)] = 1

        for e in layers[2]:
            label_map[range(*e)] = -2

        for e in layers[3]:
            label_map[range(*e)] = 2

        return label_map

    def split_map_by_marker(self, label_map, marker: int) -> npt.NDArray:
        events = np.split(
            np.where(label_map == marker)[0],
            np.where(np.diff(np.where(label_map == marker)[0]) != 1)[0] + 1,
        )
        try:
            events = np.vstack([(e[0], e[-1] + 1) for e in events])
        except ValueError and IndexError:
            events = np.array([], dtype=np.int32)

        return events

    def split_map(
        self, label_map
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        neg_events = self.split_map_by_marker(label_map, marker=0)
        actual_pos_events = self.split_map_by_marker(label_map, marker=2)
        pred_pos_events = self.split_map_by_marker(label_map, marker=1)
        soft_gray_zones = self.split_map_by_marker(label_map, marker=-1)
        hard_gray_zones = self.split_map_by_marker(label_map, marker=-2)

        return (
            neg_events,
            actual_pos_events,
            pred_pos_events,
            soft_gray_zones,
            hard_gray_zones,
        )

    def label_event(self, arr, config) -> npt.NDArray:
        actual_pos_event = self.mark_actual_pos_event(config, arr, grp_rng=0)
        soft_gray_zone, hard_gray_zone = self.mark_gray_zone(
            config, arr, actual_pos_event
        )
        pred_pos_event = self.mark_predicted_pos_event(config, actual_pos_event)
        label_map = self.mark_on_label_map(
            arr,
            layers=[
                soft_gray_zone,
                pred_pos_event,
                hard_gray_zone,
                actual_pos_event,
            ],
        )
        event_package = self.split_map(label_map)

        return event_package

    def extract_pos_segment(self, events, config):
        fs = config["fs"]
        input_len = config["input_len"] * fs

        try:
            pos_seg = np.vstack(
                [
                    sliding_window_view(np.arange(*e), input_len)[::input_len]
                    for e in events
                ]
            )
            pos_segments = np.vstack([(e[0], e[-1] + 1) for e in pos_seg])
        except ValueError or IndexError:
            # print("No positive segments extracted.")
            pos_segments = np.array([], dtype=np.int32)

        return pos_segments

    def extract_neg_segment(self, events, config):
        fs = config["fs"]
        input_len = config["input_len"] * fs
        stride = config["segment"]["seg_stride"] * fs * config["segment"]["seg_density"]

        try:
            neg_seg = np.vstack(
                [sliding_window_view(np.arange(*e), input_len)[::stride] for e in events]
            )
            neg_segments = np.vstack([(e[0], e[-1] + 1) for e in neg_seg])
        except ValueError or IndexError:
            # print("No negative segments extracted.")
            neg_segments = np.array([], dtype=np.int32)
        
        return neg_segments

    def extract_segment(self, events, config, label):
        if label == "pos":
            segments = self.extract_pos_segment(events.copy(), config)
        elif label == "neg":
            segments = self.extract_neg_segment(events.copy(), config)
        else:
            print("Not implemented.")
        return segments


class HypophetVal1(Hypophet):
    """
    Hypophet 라벨링 전략 + Acumen 샘플링 전략
    """

    name = "hypophetval1"

    def __init__(self):
        super().__init__()

    def extract_pos_segment(self, events, config):
        input_len = config["fs"] * config["input_len"]
        pos_len = config["fs"] * config["pred_len"]
        try:
            # seg_end_idx = (events[:, 0] - pos_len).reshape(-1, 1)
            # 이미 라벨링 단계에서 15분전으로 이동했으므로
            seg_end_idx = events[:, 0].reshape(-1, 1)
            seg_start_idx = seg_end_idx - input_len
            seg_range = np.hstack([seg_start_idx, seg_end_idx])
            seg_range = np.vstack([seg_range])
            seg_range = seg_range[np.where(seg_range[:, 0] >= 0)]
        except Exception as e:
            seg_range = np.array([], dtype=np.int32)
        return seg_range

    def extract_neg_segment(self, events, config):
        input_len = config["fs"] * config["input_len"]
        try:
            seg_end_idx = (events.sum(axis=-1) // 2).reshape(-1, 1)
            seg_start_idx = seg_end_idx - input_len
            seg_range = np.hstack([seg_start_idx, seg_end_idx])
            seg_range = np.vstack([seg_range])
            seg_range = seg_range[np.where(seg_range[:, 0] >= 0)]
        except Exception as e:
            seg_range = np.array([], dtype=np.int32)
        return seg_range

    def extract_segment(self, events, config, label):
        if label == "pos":
            seg_range = self.extract_pos_segment(events.copy(), config)
        elif label == "neg":
            seg_range = self.extract_neg_segment(events.copy(), config)
        else:
            print("Not implemented.")
        return seg_range


class HypophetVal2(Hypophet):
    """
    Hypophet 라벨링 전략 + Asan 샘플링 전략
    """

    name = "hypophetval2"

    def __init__(self):
        super().__init__()

    def extract_pos_segment(self, events, config):
        input_len = config["fs"] * config["input_len"]  # 30s
        pos_len = config["fs"] * config["pred_len"]  # 15m
        try:
            # seg_end_idx = (events[:, 0] - pos_len).reshape(-1, 1)
            # 이미 라벨링 단계에서 15분전으로 이동했으므로
            seg_end_idx = events[:, 0].reshape(-1, 1)
            seg_start_idx = seg_end_idx - input_len
            seg_range = np.hstack([seg_start_idx, seg_end_idx])
            seg_range = np.vstack([seg_range])
            seg_range = seg_range[np.where(seg_range[:, 0] >= 0)]
        except Exception as e:
            seg_range = np.array([], dtype=np.int32)
        return seg_range

    def extract_neg_segment(self, events, config):
        input_len = config["fs"] * config["input_len"]
        stride = (
            config["segment"]["seg_stride"]
            * config["fs"]
            * config["segment"]["seg_density"]
        )
        try:
            neg_seg = []
            for e in events:
                seg = sliding_window_view(range(*e), input_len)[::stride]
                if len(seg) > 30:
                    rand_idx = np.random.choice(
                        np.arange(len(seg)), size=30, replace=False
                    )
                    seg = seg[rand_idx]
                neg_seg.append(seg)
            neg_seg = np.vstack(neg_seg)
            seg_range = np.hstack([neg_seg[:, 0:1], neg_seg[:, -1:] + 1])
        except Exception as e:
            seg_range = np.array([], dtype=np.int32)
        return seg_range

    def extract_segment(self, events, config, label):
        np.random.seed(config["seed"])
        if label == "pos":
            seg_range = self.extract_pos_segment(events.copy(), config)
        elif label == "neg":
            seg_range = self.extract_neg_segment(events.copy(), config)
        else:
            print("Not implemented.")
        return seg_range


class HypophetVal3(Hypophet):
    """
    Hypophet 라벨링 전략 + Moghadam 샘플링 전략
    """

    name = "hypophetval3"

    def __init__(self):
        super().__init__()

    def extract_pos_segment(self, events, config):
        if len(events) == 0:
            return np.array([], dtype=np.int32)
        input_size = config["fs"] * config["input_len"]
        win_size = config["fs"] * config["pred_len"]

        start_idx = (events[:, 0] - input_size).reshape(-1, 1)
        end_idx = start_idx + win_size
        start_idx = np.where(start_idx < 0, 0, start_idx)
        seg_range = np.hstack([start_idx, end_idx])
        seg_range = np.vstack([seg_range])
        event_idx = [np.arange(start, end) for start, end in seg_range]

        pos_segment = []
        for ei in event_idx:
            pos_segment.append(sliding_window_view(ei, input_size)[::input_size, :])

        pos_segment = np.vstack(pos_segment)
        pos_segment = np.unique(pos_segment, axis=0)
        return np.hstack([pos_segment[:, 0:1], pos_segment[:, -1:] + 1])

    def extract_neg_segment(self, events, config):
        if len(events) == 0:
            return np.array([], dtype=np.int32)
        input_size = config["fs"] * config["input_len"]
        neg_segment = []
        for e in [np.arange(start, end) for start, end in events]:
            neg_segment.append(sliding_window_view(e, input_size)[::input_size])
        neg_segment = np.vstack(neg_segment)
        neg_segment = neg_segment[:, [0, -1]]
        rand_idx = np.random.choice(
            range(len(neg_segment)), round(len(neg_segment) * 0.2)
        )
        return neg_segment[rand_idx]

    def extract_segment(self, events, config, label):
        np.random.seed(config["seed"])
        if label == "pos":
            seg_range = self.extract_pos_segment(events.copy(), config)
        elif label == "neg":
            seg_range = self.extract_neg_segment(events.copy(), config)
        else:
            print("Not implemented.")
        return seg_range


class Acumen(LabelStrategy):
    name = "acumen"

    def get_segment(self, condition) -> npt.NDArray:
        """
        Input:  MAP 배열
        Return: 입력으로부터 추출한 모든 영역의 [start, end]
        Description:
            조건에 부합하는 지점들을 찾는다.
            right_shift 는 현재 기준 직전 MAP 값, left_shift 는 현재 기준 직후 MAP 값 이다.
            is_pos_start 는 추출된 모든 영역의 start idx 를 갖는다.
            is_pos_end 는 추출된 모든 영역의 end idx 를 갖는다.
            항상 start 는 start idx, end 는 end idx 를 갖게 하기 위해 constant_values 를
            right_shift, left_shift 에서 모두 0으로 두었다.
            이러면 `np.where((is_hypo ^ right_shift) == True)[0]` 은 항상 start 로 시작하고,
            마지막은 항상 end 로 끝날 것이다.
            각 영역의 [start, end] 를 넘파이 배열로 구성한다.
            `np.arange` 에 바로 입력하기 용이하도록 end 에 +1 을 해준다.
            (사실 코딩만 따졌을 때는 더 쉽게 할 수도 있지만, 가독성을 높이기 위해 이와 같이 진행하였다.
            연산속도에는 큰 차이가 없어 보인다.)
        """
        is_hypo = condition
        right_shift = np.pad(is_hypo[:-1], (1, 0), mode="constant", constant_values=0)
        left_shift = np.pad(is_hypo[1:], (0, 1), mode="constant", constant_values=0)
        is_pos_start = np.where((is_hypo ^ right_shift) == True)[0][::2]
        is_pos_end = np.where((is_hypo ^ left_shift) == True)[0][1::2]
        raw_pos_event = np.array(list(zip(is_pos_start, is_pos_end)))
        if len(raw_pos_event) != 0:
            raw_pos_event[:, 1] = raw_pos_event[:, 1] + 1
        return raw_pos_event

    def extract_segment_at_least(self, events, threshold) -> npt.NDArray:
        """
        Description:
            events 는 [start idx, end idx] 로 구성된 넘파이 배열이다.
            events 중 길이가 threshold 이상인 것만 추출한다.
        """
        if len(events) != 0:
            result = events[np.where(np.diff(events, axis=1) >= threshold)[0]]
        else:
            result = events
        return result

    def filter_fast_drop(self, arr, events, shift):
        if len(events) == 0:
            return np.array([], dtype=np.int32)

        is_fast_drop = []
        for e in events:
            pos_value = arr[np.arange(*e)]
            pos_value_1sec = np.pad(pos_value[shift - 1 :], (0, shift - 1), mode="edge")
            pos_value_1min = np.pad(
                pos_value[(shift * 60) - 1 :], (0, (shift * 60) - 1), mode="edge"
            )
            delta_1sec = pos_value - pos_value_1sec
            delta_1min = pos_value - pos_value_1min
            is_fast_drop.append(np.any((delta_1sec >= 0.5) & (delta_1min >= 30)))

        is_fast_drop = np.array(is_fast_drop)
        return events[~is_fast_drop]

    def filter_overlap(self, events, pos_event, adj):
        """
        adj 만큼 확장된 pos_event 에 대해서는 두 가지 경우만 가능하다.
        1. pos_event 안에 events의 start 가 overlap 되거나
        2. pos_event 안에 events의 end 가 overlap 되거나
        """
        if len(events) == 0:
            return np.array([], dtype=np.int64)

        if len(pos_event) != 0:
            pos_event[:, 0] = pos_event[:, 0] - adj
            pos_event[:, 1] = pos_event[:, 1] + adj
            is_overlap = []
            for start, end in events:
                is_start_overlap = (pos_event[:, 0] <= start) & (
                    pos_event[:, 1] > start
                )
                is_end_overlap = (pos_event[:, 0] < end) & (pos_event[:, 1] >= end)
                is_overlap.append(np.any(is_start_overlap | is_end_overlap))
            is_overlap = np.array(is_overlap)
            return events[~is_overlap]

        else:
            return events

    def label_pos_event(self, arr, config) -> npt.NDArray:
        """
        1. MAP < 65 인 지점을 전부 가져온다.
        2. 지점들이 1분 이상 연속되어 영역을 만드는 경우만 추출한다.
        3. 해당 영역 내에서 갑작스런 하강이 존재하는 경우를 필터링한다. (>= 0.5 mmHg/s, >= 30 mmHg/min)
        """
        raw_pos_event = self.get_segment((arr < 65) & (arr > 0))
        raw_pos_event = self.extract_segment_at_least(raw_pos_event, config["fs"] * 60)
        pos_event = self.filter_fast_drop(arr, raw_pos_event, config["fs"])
        return pos_event

    def label_neg_event(self, arr, pos_event, config):
        """
        1. MAP > 75 인 영역을 전부 가져온다.
        2. 지점들이 30분 이상 연속되어 영역을 만드는 경우만 추출한다.
        3. 해당 영역에 인접한 저혈압 영역들로부터 20분 이상 이격되지 못한 경우를 필터링한다.
        """
        raw_neg_event = self.get_segment((arr > 75))
        raw_neg_event = self.extract_segment_at_least(
            raw_neg_event, config["fs"] * 60 * 30
        )
        neg_event = self.filter_overlap(
            raw_neg_event, pos_event.copy(), config["fs"] * 60 * 20
        )
        return neg_event

    def extract_pos_segment(self, events, config):
        input_len = config["fs"] * config["input_len"]
        pos_len = config["fs"] * config["pred_len"]
        try:
            seg_end_idx = (events[:, 0] - pos_len).reshape(-1, 1)
            seg_start_idx = seg_end_idx - input_len
            seg_range = np.hstack([seg_start_idx, seg_end_idx])
            seg_range = np.vstack([seg_range])
            seg_range = seg_range[np.where(seg_range[:, 0] >= 0)]
        except Exception as e:
            seg_range = np.array([], dtype=np.int32)
        return seg_range

    def extract_neg_segment(self, events, config):
        input_len = config["fs"] * config["input_len"]
        try:
            seg_end_idx = (events.sum(axis=-1) // 2).reshape(-1, 1)
            seg_start_idx = seg_end_idx - input_len
            seg_range = np.hstack([seg_start_idx, seg_end_idx])
            seg_range = np.vstack([seg_range])
            seg_range = seg_range[np.where(seg_range[:, 0] >= 0)]
        except Exception as e:
            seg_range = np.array([], dtype=np.int32)
        return seg_range

    def extract_segment(self, events, config, label):
        if label == "pos":
            seg_range = self.extract_pos_segment(events.copy(), config)
        elif label == "neg":
            seg_range = self.extract_neg_segment(events.copy(), config)
        else:
            print("Not implemented.")
        return seg_range


class Acumen2(LabelStrategy):
    """
    Acumen 전략에서 샘플링 전략만 Hypophet 방식으로 바꾼 것.
    Acumen에서 음성 샘플링 전략만 바꾸더라도 모델이 보다 다양한 비저혈압 패턴을 학습할 것이므로
    이전보다 임상적 유용성이 개선될 수 있다.
    """

    name = "acumen2"

    def get_segment(self, condition) -> npt.NDArray:
        """
        Input:  MAP 배열
        Return: 입력으로부터 추출한 모든 영역의 [start, end]
        Description:
            조건에 부합하는 지점들을 찾는다.
            right_shift 는 현재 기준 직전 MAP 값, left_shift 는 현재 기준 직후 MAP 값 이다.
            is_pos_start 는 추출된 모든 영역의 start idx 를 갖는다.
            is_pos_end 는 추출된 모든 영역의 end idx 를 갖는다.
            항상 start 는 start idx, end 는 end idx 를 갖게 하기 위해 constant_values 를
            right_shift, left_shift 에서 모두 0으로 두었다.
            이러면 `np.where((is_hypo ^ right_shift) == True)[0]` 은 항상 start 로 시작하고,
            마지막은 항상 end 로 끝날 것이다.
            각 영역의 [start, end] 를 넘파이 배열로 구성한다.
            `np.arange` 에 바로 입력하기 용이하도록 end 에 +1 을 해준다.
            (사실 코딩만 따졌을 때는 더 쉽게 할 수도 있지만, 가독성을 높이기 위해 이와 같이 진행하였다.
            연산속도에는 큰 차이가 없어 보인다.)
        """
        is_hypo = condition
        right_shift = np.pad(is_hypo[:-1], (1, 0), mode="constant", constant_values=0)
        left_shift = np.pad(is_hypo[1:], (0, 1), mode="constant", constant_values=0)
        is_pos_start = np.where((is_hypo ^ right_shift) == True)[0][::2]
        is_pos_end = np.where((is_hypo ^ left_shift) == True)[0][1::2]
        raw_pos_event = np.array(list(zip(is_pos_start, is_pos_end)))
        if len(raw_pos_event) != 0:
            raw_pos_event[:, 1] = raw_pos_event[:, 1] + 1
        return raw_pos_event

    def extract_segment_at_least(self, events, threshold) -> npt.NDArray:
        """
        Description:
            events 는 [start idx, end idx] 로 구성된 넘파이 배열이다.
            events 중 길이가 threshold 이상인 것만 추출한다.
        """
        if len(events) != 0:
            result = events[np.where(np.diff(events, axis=1) >= threshold)[0]]
        else:
            result = events
        return result

    def filter_fast_drop(self, arr, events, shift):
        if len(events) == 0:
            return np.array([], dtype=np.int32)

        is_fast_drop = []
        for e in events:
            pos_value = arr[np.arange(*e)]
            pos_value_1sec = np.pad(pos_value[shift - 1 :], (0, shift - 1), mode="edge")
            pos_value_1min = np.pad(
                pos_value[(shift * 60) - 1 :], (0, (shift * 60) - 1), mode="edge"
            )
            delta_1sec = pos_value - pos_value_1sec
            delta_1min = pos_value - pos_value_1min
            is_fast_drop.append(np.any((delta_1sec >= 0.5) & (delta_1min >= 30)))

        is_fast_drop = np.array(is_fast_drop)
        return events[~is_fast_drop]

    def filter_overlap(self, events, pos_event, adj):
        """
        adj 만큼 확장된 pos_event 에 대해서는 두 가지 경우만 가능하다.
        1. pos_event 안에 events의 start 가 overlap 되거나
        2. pos_event 안에 events의 end 가 overlap 되거나
        """
        if len(events) == 0:
            return np.array([], dtype=np.int64)

        if len(pos_event) != 0:
            pos_event[:, 0] = pos_event[:, 0] - adj
            pos_event[:, 1] = pos_event[:, 1] + adj
            is_overlap = []
            for start, end in events:
                is_start_overlap = (pos_event[:, 0] <= start) & (
                    pos_event[:, 1] > start
                )
                is_end_overlap = (pos_event[:, 0] < end) & (pos_event[:, 1] >= end)
                is_overlap.append(np.any(is_start_overlap | is_end_overlap))
            is_overlap = np.array(is_overlap)
            return events[~is_overlap]

        else:
            return events

    def label_pos_event(self, arr, config) -> npt.NDArray:
        """
        1. MAP < 65 인 지점을 전부 가져온다.
        2. 지점들이 1분 이상 연속되어 영역을 만드는 경우만 추출한다.
        3. 해당 영역 내에서 갑작스런 하강이 존재하는 경우를 필터링한다. (>= 0.5 mmHg/s, >= 30 mmHg/min)
        """
        raw_pos_event = self.get_segment((arr < 65) & (arr > 0))
        raw_pos_event = self.extract_segment_at_least(raw_pos_event, config["fs"] * 60)
        pos_event = self.filter_fast_drop(arr, raw_pos_event, config["fs"])
        return pos_event

    def label_neg_event(self, arr, pos_event, config):
        """
        1. MAP > 75 인 영역을 전부 가져온다.
        2. 지점들이 30분 이상 연속되어 영역을 만드는 경우만 추출한다.
        3. 해당 영역에 인접한 저혈압 영역들로부터 20분 이상 이격되지 못한 경우를 필터링한다.
        """
        raw_neg_event = self.get_segment((arr > 75))
        raw_neg_event = self.extract_segment_at_least(
            raw_neg_event, config["fs"] * 60 * 30
        )
        neg_event = self.filter_overlap(
            raw_neg_event, pos_event.copy(), config["fs"] * 60 * 20
        )
        return neg_event

    def extract_pos_segment(self, events, config):
        input_len = config["fs"] * config["input_len"]
        pos_len = config["fs"] * config["pred_len"]
        try:
            seg_end_idx = (events[:, 0] - pos_len).reshape(-1, 1)
            seg_start_idx = seg_end_idx - input_len
            seg_range = np.hstack([seg_start_idx, seg_end_idx])
            seg_range = np.vstack([seg_range])
            seg_range = seg_range[np.where(seg_range[:, 0] >= 0)]
        except Exception as e:
            seg_range = np.array([], dtype=np.int32)
        return seg_range

    def extract_neg_segment(self, events, config):
        fs = config["fs"]
        input_len = config["input_len"] * fs
        # TODO: 만약 moghadam 처럼 input_len 간격으로 샘플링을 한다면? -> 딱히 차이 없다.
        stride = config["segment"]["seg_stride"] * fs * config["segment"]["seg_density"]
        # stride = input_len

        if len(events) == 0:
            return np.array([], dtype=np.int32)

        seg_idx = [sliding_window_view(range(*e), input_len)[::stride] for e in events]
        seg_idx = np.vstack(seg_idx)
        seg_range = np.hstack([seg_idx[:, 0:1], seg_idx[:, -1:] + 1])
        return seg_range

    def extract_segment(self, events, config, label):
        if label == "pos":
            seg_range = self.extract_pos_segment(events.copy(), config)
        elif label == "neg":
            seg_range = self.extract_neg_segment(events.copy(), config)
        else:
            print("Not implemented.")
        return seg_range


class Asan(LabelStrategy):
    name = "asan"

    def get_segment(self, condition) -> npt.NDArray:
        """
        Input:  MAP 배열
        Return: 입력으로부터 추출한 모든 영역의 [start, end]
        Description:
            조건에 부합하는 지점들을 찾는다.
            right_shift 는 현재 기준 직전 MAP 값, left_shift 는 현재 기준 직후 MAP 값 이다.
            is_pos_start 는 추출된 모든 영역의 start idx 를 갖는다.
            is_pos_end 는 추출된 모든 영역의 end idx 를 갖는다.
            항상 start 는 start idx, end 는 end idx 를 갖게 하기 위해 constant_values 를
            right_shift, left_shift 에서 모두 0으로 두었다.
            이러면 `np.where((is_hypo ^ right_shift) == True)[0]` 은 항상 start 로 시작하고,
            마지막은 항상 end 로 끝날 것이다.
            각 영역의 [start, end] 를 넘파이 배열로 구성한다.
            `np.arange` 에 바로 입력하기 용이하도록 end 에 +1 을 해준다.
            (사실 코딩만 따졌을 때는 더 쉽게 할 수도 있지만, 가독성을 높이기 위해 이와 같이 진행하였다.
            연산속도에는 큰 차이가 없어 보인다.)
        """
        is_hypo = condition
        right_shift = np.pad(is_hypo[:-1], (1, 0), mode="constant", constant_values=0)
        left_shift = np.pad(is_hypo[1:], (0, 1), mode="constant", constant_values=0)
        is_pos_start = np.where((is_hypo ^ right_shift) == True)[0][::2]
        is_pos_end = np.where((is_hypo ^ left_shift) == True)[0][1::2]
        raw_pos_event = np.array(list(zip(is_pos_start, is_pos_end)))
        if len(raw_pos_event) != 0:
            raw_pos_event[:, 1] = raw_pos_event[:, 1] + 1
        return raw_pos_event

    def extract_segment_at_least(self, events, threshold) -> npt.NDArray:
        """
        Description:
            events 는 [start idx, end idx] 로 구성된 넘파이 배열이다.
            events 중 길이가 threshold 이상인 것만 추출한다.
        """
        if len(events) != 0:
            result = events[np.where(np.diff(events, axis=1) >= threshold)[0]]
        else:
            result = events
        return result

    def label_pos_event(self, arr, config) -> npt.NDArray:
        raw_pos_event = self.get_segment((arr < 65) & (arr > 0))
        pos_event = self.extract_segment_at_least(raw_pos_event, config["fs"] * 60)
        return pos_event

    def label_neg_event(self, arr, pos_event, config) -> npt.NDArray:
        raw_neg_event = self.get_segment((arr > 75))
        neg_event = self.extract_segment_at_least(raw_neg_event, config["fs"] * 60)
        return neg_event

    def extract_pos_segment(self, events, config):
        input_len = config["fs"] * config["input_len"]  # 30s
        pos_len = config["fs"] * config["pred_len"]  # 15m
        try:
            seg_end_idx = (events[:, 0] - pos_len).reshape(-1, 1)
            seg_start_idx = seg_end_idx - input_len
            seg_range = np.hstack([seg_start_idx, seg_end_idx])
            seg_range = np.vstack([seg_range])
            seg_range = seg_range[np.where(seg_range[:, 0] >= 0)]
        except Exception as e:
            seg_range = np.array([], dtype=np.int32)
        return seg_range

    def extract_neg_segment(self, events, config):
        input_len = config["fs"] * config["input_len"]
        stride = (
            config["segment"]["seg_stride"]
            * config["fs"]
            * config["segment"]["seg_density"]
        )
        try:
            neg_seg = []
            for e in events:
                seg = sliding_window_view(range(*e), input_len)[::stride]
                if len(seg) > 30:
                    rand_idx = np.random.choice(
                        np.arange(len(seg)), size=30, replace=False
                    )
                    seg = seg[rand_idx]
                neg_seg.append(seg)
            neg_seg = np.vstack(neg_seg)
            seg_range = np.hstack([neg_seg[:, 0:1], neg_seg[:, -1:] + 1])
        except Exception as e:
            seg_range = np.array([], dtype=np.int32)
        return seg_range

    def extract_segment(self, events, config, label):
        np.random.seed(config["seed"])
        if label == "pos":
            seg_range = self.extract_pos_segment(events.copy(), config)
        elif label == "neg":
            seg_range = self.extract_neg_segment(events.copy(), config)
        else:
            print("Not implemented.")
        return seg_range


class Moghadam(LabelStrategy):
    name = "moghadam"

    def find_pos_onset(self, arr, config):
        input_size = config["fs"] * config["input_len"]
        win_size = config["fs"] * config["pred_len"]
        if len(arr) < win_size:
            return np.array([], dtype=np.int32)
        windows_idx = sliding_window_view(range(len(arr)), win_size)[::input_size]
        windows = sliding_window_view(arr, win_size)[::input_size]
        hypo_percentage = np.sum(windows, axis=1, keepdims=True) / windows.shape[1]
        hypo_onset = windows_idx[np.where(hypo_percentage >= 0.9)[0]]
        pos_event = hypo_onset[:, [0, -1]]
        pos_event[:, -1] = pos_event[:, -1] + 1
        return pos_event

    def mark_influence(self, arr, event, config):
        is_influence = np.zeros(len(arr)).astype(bool)
        if len(event) == 0:
            return is_influence
        event[:, 0] = event[:, 0] - config["fs"] * 60 * 40
        event[:, 1] = event[:, 1] + config["fs"] * 60 * 20
        event = np.where(event < 0, 0, event)
        event = np.where(event >= len(arr), len(arr), event)
        influence_idx = np.unique(np.hstack([np.arange(s, e) for s, e in event]))
        is_influence[influence_idx] = True
        return is_influence

    def get_segment(self, cond) -> npt.NDArray:
        """
        Input:  MAP 배열
        Return: 입력으로부터 추출한 모든 영역의 [start, end]
        Description:
            조건에 부합하는 지점들을 찾는다.
            right_shift 는 현재 기준 직전 MAP 값, left_shift 는 현재 기준 직후 MAP 값 이다.
            is_pos_start 는 추출된 모든 영역의 start idx 를 갖는다.
            is_pos_end 는 추출된 모든 영역의 end idx 를 갖는다.
            항상 start 는 start idx, end 는 end idx 를 갖게 하기 위해 constant_values 를
            right_shift, left_shift 에서 모두 0으로 두었다.
            이러면 `np.where((is_hypo ^ right_shift) == True)[0]` 은 항상 start 로 시작하고,
            마지막은 항상 end 로 끝날 것이다.
            각 영역의 [start, end] 를 넘파이 배열로 구성한다.
            `np.arange` 에 바로 입력하기 용이하도록 end 에 +1 을 해준다.
            (사실 코딩만 따졌을 때는 더 쉽게 할 수도 있지만, 가독성을 높이기 위해 이와 같이 진행하였다.
            연산속도에는 큰 차이가 없어 보인다.)
        """
        right_shift = np.pad(cond[:-1], (1, 0), mode="constant", constant_values=0)
        left_shift = np.pad(cond[1:], (0, 1), mode="constant", constant_values=0)
        is_start = np.where((cond ^ right_shift) == True)[0][::2]
        is_end = np.where((cond ^ left_shift) == True)[0][1::2]
        raw_event = np.array(list(zip(is_start, is_end)))
        if len(raw_event) != 0:
            raw_event[:, 1] = raw_event[:, 1] + 1
        return raw_event

    def label_pos_event(self, arr, config):
        """
        * 1. MAP < 65 인 데이터포인트를 전체 신호에서 찾는다.
        * 2. 위 조건이 True 인 경우가 15분 동안 90% 이상인 경우를 저혈압 이벤트로 추출한다.
        """
        pos_event = self.find_pos_onset((arr < 65) & (arr > 0), config)
        return pos_event

    def label_neg_event(self, arr, pos_event, config):
        """
        * 1. MAP > 75 인 데이터포인트를 전체 신호에서 찾는다.
        * 2. 단, 저혈압 이벤트의 시작지점으로부터는 40분, 종료지점으로부터는 20분 이격되어야 한다.
        """
        is_normo = arr > 75
        is_hypo_influence = self.mark_influence(arr, pos_event.copy(), config)
        raw_neg_event = self.get_segment(is_normo & ~is_hypo_influence)
        if len(raw_neg_event) == 0:
            return np.array([], dtype=np.int32)
        is_at_least_input_len = np.where(
            np.diff(raw_neg_event, axis=1) >= (config["fs"] * config["input_len"])
        )[0]
        return raw_neg_event[is_at_least_input_len]

    def extract_pos_segment(self, events, config):
        if len(events) == 0:
            return np.array([], dtype=np.int32)
        input_size = config["fs"] * config["input_len"]
        win_size = config["fs"] * config["pred_len"]
        event_idx = np.vstack([np.arange(start, end) for start, end in events])
        pos_segment = np.vstack(
            sliding_window_view(event_idx - win_size, input_size, axis=1)[
                :, ::input_size, :
            ]
        )
        pos_segment = np.unique(pos_segment, axis=0)
        return pos_segment[:, [0, -1]]

    def extract_neg_segment(self, events, config):
        if len(events) == 0:
            return np.array([], dtype=np.int32)
        input_size = config["fs"] * config["input_len"]
        neg_segment = []
        for e in [np.arange(start, end) for start, end in events]:
            neg_segment.append(sliding_window_view(e, input_size)[::input_size])
        neg_segment = np.vstack(neg_segment)
        neg_segment = neg_segment[:, [0, -1]]
        rand_idx = np.random.choice(
            range(len(neg_segment)), round(len(neg_segment) * 0.2)
        )
        return neg_segment[rand_idx]

    def extract_segment(self, events, config, label):
        np.random.seed(config["seed"])
        if label == "pos":
            seg_range = self.extract_pos_segment(events.copy(), config)
        elif label == "neg":
            seg_range = self.extract_neg_segment(events.copy(), config)
        else:
            print("Not implemented.")
        return seg_range


class GroundTruth(LabelStrategy):
    name = "groundtruth"

    def get_segment(self, condition) -> npt.NDArray:
        """
        Input:  MAP 배열
        Return: 입력으로부터 추출한 모든 영역의 [start, end]
        Description:
            조건에 부합하는 지점들을 찾는다.
            right_shift 는 현재 기준 직전 MAP 값, left_shift 는 현재 기준 직후 MAP 값 이다.
            is_pos_start 는 추출된 모든 영역의 start idx 를 갖는다.
            is_pos_end 는 추출된 모든 영역의 end idx 를 갖는다.
            항상 start 는 start idx, end 는 end idx 를 갖게 하기 위해 constant_values 를
            right_shift, left_shift 에서 모두 0으로 두었다.
            이러면 `np.where((is_hypo ^ right_shift) == True)[0]` 은 항상 start 로 시작하고,
            마지막은 항상 end 로 끝날 것이다.
            각 영역의 [start, end] 를 넘파이 배열로 구성한다.
            `np.arange` 에 바로 입력하기 용이하도록 end 에 +1 을 해준다.
            (사실 코딩만 따졌을 때는 더 쉽게 할 수도 있지만, 가독성을 높이기 위해 이와 같이 진행하였다.
            연산속도에는 큰 차이가 없어 보인다.)
        """
        is_hypo = condition
        right_shift = np.pad(is_hypo[:-1], (1, 0), mode="constant", constant_values=0)
        left_shift = np.pad(is_hypo[1:], (0, 1), mode="constant", constant_values=0)
        is_pos_start = np.where((is_hypo ^ right_shift) == True)[0][::2]
        is_pos_end = np.where((is_hypo ^ left_shift) == True)[0][1::2]
        raw_pos_event = np.array(list(zip(is_pos_start, is_pos_end)))
        if len(raw_pos_event) != 0:
            raw_pos_event[:, 1] = raw_pos_event[:, 1] + 1
        return raw_pos_event

    def extract_segment_at_least(self, events, threshold) -> npt.NDArray:
        """
        Description:
            events 는 [start idx, end idx] 로 구성된 넘파이 배열이다.
            events 중 길이가 threshold 이상인 것만 추출한다.
        """
        if len(events) != 0:
            result = events[np.where(np.diff(events, axis=1) >= threshold)[0]]
        else:
            result = events
        return result

    def extend_to_past(self, events, past_len) -> npt.NDArray:
        past_loc = events[:, 0:1] - past_len
        start = np.where(past_loc < 0, 0, past_loc)
        end = events[:, -1:]
        return np.hstack([start, end], dtype=np.int32)

    def group_event(self, events, config) -> npt.NDArray:
        """
        * Make range for grouping for each event.
        * Events intersects with each others get a group number.
        *
        * For example, if segment A's range is (1200, 2500) and segment B's range is (2400, 3000),
        * each segment is intersect at (2400, 2500).
        * Thus they can be merged into new segment with range (1200, 3000).
        """
        event_cond = config["event_condition"]["pos"]
        grp_rng = event_cond["group_range"]

        grp_len = config["fs"] * 60 * grp_rng
        events_with_grp_rng = [
            (event[0] - grp_len, event[-1] + grp_len) for event in events
        ]

        group_no = np.zeros(len(events_with_grp_rng))
        for i in range(len(events_with_grp_rng) - 1):
            segment = events_with_grp_rng[i]
            next_segment = events_with_grp_rng[i + 1]
            start, end = segment[0], segment[-1]

            if start <= next_segment[0] < end:
                group_no[i + 1] = group_no[i]
            else:
                group_no[i + 1] = group_no[i] + 1

        events_with_grp = tuple(zip(events, group_no))
        grp_events = []
        for _, grp in groupby(events_with_grp, lambda x: x[1]):
            temp_grp = list(grp)
            start_idx, end_idx = temp_grp[0][0][0], temp_grp[-1][0][-1]
            grp_events.append((start_idx, end_idx))
        return np.vstack(grp_events, dtype=np.int32)

    def label_pos_event(self, arr, config):
        fs = config["fs"]
        input_len, pred_len = config["input_len"] * fs, config["pred_len"] * fs
        raw_pos_event = self.get_segment((arr < 65) & (arr > 0))
        pos_event = self.extract_segment_at_least(raw_pos_event, config["fs"] * 60)
        if len(pos_event) != 0:
            # pos_event[:, 0] = pos_event[:, 0] - (input_len + pred_len)
            # pos_event = np.where(pos_event < 0, 0, pos_event)

            # Updated 26.01.06
            # 위 코드의 경우 겹치는 이벤트들을 하나의 이벤트로 처리해주지 않았다.
            # 아래 코드를 사용하면 겹치는 이벤트를 하나의 이벤트로 만들어준다.

            pos_event = self.extend_to_past(pos_event, input_len + pred_len)
            pos_event = self.group_event(pos_event, config)  # grp_range = 0
            # 이렇게 하면 두 이벤트가 겹치는 경우는 하나로 처리하게 됨.
        return pos_event

    def label_neg_event(self, arr, pos_event, config):
        if len(pos_event) != 0:
            pos_idx = np.hstack([np.arange(*e) for e in pos_event])
        else:
            pos_idx = np.array([], dtype=np.int32)

        raw_idx = np.arange(arr.size)
        try:
            neg_idx = np.setdiff1d(raw_idx, pos_idx)
            split_loc = np.where(np.diff(neg_idx, prepend=neg_idx[0] - 1) != 1)[0]
            split_neg_idx = np.split(neg_idx, split_loc)
            neg_event = np.vstack([e[[0, -1]] for e in split_neg_idx])

        except Exception as e:
            neg_event = np.array([], dtype=np.int32)

        return neg_event

    def extract_segment(self, events, config, label):
        if len(events) == 0:
            return np.array([], dtype=np.int32)

        input_len = config["fs"] * config["input_len"]

        events = events[np.where(np.diff(events, axis=1) >= input_len)[0]]

        seg_idx = [
            sliding_window_view(range(*e), input_len)[::input_len] for e in events
        ]
        seg_idx = np.vstack(seg_idx)
        seg_range = np.hstack([seg_idx[:, 0:1], seg_idx[:, -1:] + 1])
        return seg_range


class GroundTruth2(LabelStrategy):
    name = "groundtruth2"

    def set_window_of(self, arr, win_size) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Windowing input array with constant win_size.
        """
        if len(arr) < win_size:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
        windows = sliding_window_view(arr, window_shape=win_size)[::win_size]
        windows_idx = np.arange(windows.size).reshape(-1, win_size)
        # end point 에 1을 더하여 이후 배열을 생성할 때 용이하도록 한다.
        windows_range = np.hstack([windows_idx[:, 0:1], windows_idx[:, -1:] + 1])
        return windows, windows_range

    def extend_to_past(self, events, past_len) -> npt.NDArray:
        past_loc = events[:, 0:1] - past_len
        start = np.where(past_loc < 0, 0, past_loc)
        end = events[:, -1:]
        return np.hstack([start, end], dtype=np.int32)

    def group_event(self, events, config) -> npt.NDArray:
        """
        * Make range for grouping for each event.
        * Events intersects with each others get a group number.
        *
        * For example, if segment A's range is (1200, 2500) and segment B's range is (2400, 3000),
        * each segment is intersect at (2400, 2500).
        * Thus they can be merged into new segment with range (1200, 3000).
        """
        event_cond = config["event_condition"]["pos"]
        grp_rng = event_cond["group_range"]

        grp_len = config["fs"] * 60 * grp_rng
        events_with_grp_rng = [
            (event[0] - grp_len, event[-1] + grp_len) for event in events
        ]

        group_no = np.zeros(len(events_with_grp_rng))
        for i in range(len(events_with_grp_rng) - 1):
            segment = events_with_grp_rng[i]
            next_segment = events_with_grp_rng[i + 1]
            start, end = segment[0], segment[-1]

            if start <= next_segment[0] < end:
                group_no[i + 1] = group_no[i]
            else:
                group_no[i + 1] = group_no[i] + 1

        events_with_grp = tuple(zip(events, group_no))
        grp_events = []
        for _, grp in groupby(events_with_grp, lambda x: x[1]):
            temp_grp = list(grp)
            start_idx, end_idx = temp_grp[0][0][0], temp_grp[-1][0][-1]
            grp_events.append((start_idx, end_idx))
        return np.vstack(grp_events, dtype=np.int32)

    def label_pos_event(self, arr, config):
        """
        Label positive event from input array based on config file.

        1. Slice array with window size.
        2. Give condition to each windows.
        3. Label positive event(step 1): avg(MAP) < 65 at least 1 minute.
        4. Label positive event(step 2): extend to past N(target predict length) minute.
        5. Label positive event(step 3): merge adjacent positive events.
        """
        fs = config["fs"]
        input_len, pred_len = config["input_len"] * fs, config["pred_len"] * fs
        win_size = config["event_condition"]["pos"]["N"] * fs
        windows, win_rng = self.set_window_of(arr, win_size)
        is_window_valid = np.all(windows != 0, axis=1)
        is_positive = windows.mean(axis=1) < 65
        try:
            pos_event = win_rng[is_window_valid & is_positive]
            pos_event = self.extend_to_past(pos_event, input_len + pred_len)
            pos_event = self.group_event(pos_event, config)  # grp_range = 0
        except Exception as e:
            pos_event = np.array([], dtype=np.int32)

        return pos_event

    def label_neg_event(self, arr, pos_event, config):
        if len(pos_event) != 0:
            pos_idx = np.hstack([np.arange(*e) for e in pos_event])
        else:
            pos_idx = np.array([], dtype=np.int32)

        raw_idx = np.arange(arr.size)
        try:
            neg_idx = np.setdiff1d(raw_idx, pos_idx)
            split_loc = np.where(np.diff(neg_idx, prepend=neg_idx[0] - 1) != 1)[0]
            split_neg_idx = np.split(neg_idx, split_loc)
            neg_event = np.vstack([e[[0, -1]] for e in split_neg_idx])

        except Exception as e:
            neg_event = np.array([], dtype=np.int32)

        return neg_event

    def extract_segment(self, events, config, label):
        if len(events) == 0:
            return np.array([], dtype=np.int32)

        input_len = config["input_len"] * config["fs"]

        events = events[np.where(np.diff(events, axis=1) >= input_len)[0]]

        seg_idx = [
            sliding_window_view(range(*e), input_len)[::input_len] for e in events
        ]
        seg_idx = np.vstack(seg_idx)
        seg_range = np.hstack([seg_idx[:, 0:1], seg_idx[:, -1:] + 1])
        return seg_range


class HypophetTest1(Hypophet):
    """
    * Negative event 기준만 바꿨음.
    *   - MAP > 75
    *   - Pos event로부터 adj 만큼 이격되어야 함.
    *   - 최소 1개의 segment는 추출할 수 있는 영역이 형성되어야 함.
    *
    * Segment 기준은 Hypophet을 그대로 사용함.
    *
    * 효과적이지 않음, 오히려 음성을 양성으로 잘못 식별함.
    * 못 씀.
    """

    name = "hypophettest1"

    def mark_influence(self, arr, event, config):
        adj = config["event_condition"]["neg"]["adjacency"]
        is_influence = np.zeros(len(arr)).astype(bool)
        if len(event) == 0:
            return is_influence
        event[:, 0] = event[:, 0] - config["fs"] * 60 * adj
        event[:, 1] = event[:, 1] + config["fs"] * 60 * adj
        event = np.where(event < 0, 0, event)
        event = np.where(event >= len(arr), len(arr), event)
        influence_idx = np.unique(np.hstack([np.arange(s, e) for s, e in event]))
        is_influence[influence_idx] = True
        return is_influence

    def get_segment(self, cond) -> npt.NDArray:
        """
        Input:  MAP 배열
        Return: 입력으로부터 추출한 모든 영역의 [start, end]
        Description:
            조건에 부합하는 지점들을 찾는다.
            right_shift 는 현재 기준 직전 MAP 값, left_shift 는 현재 기준 직후 MAP 값 이다.
            is_pos_start 는 추출된 모든 영역의 start idx 를 갖는다.
            is_pos_end 는 추출된 모든 영역의 end idx 를 갖는다.
            항상 start 는 start idx, end 는 end idx 를 갖게 하기 위해 constant_values 를
            right_shift, left_shift 에서 모두 0으로 두었다.
            이러면 `np.where((is_hypo ^ right_shift) == True)[0]` 은 항상 start 로 시작하고,
            마지막은 항상 end 로 끝날 것이다.
            각 영역의 [start, end] 를 넘파이 배열로 구성한다.
            `np.arange` 에 바로 입력하기 용이하도록 end 에 +1 을 해준다.
            (사실 코딩만 따졌을 때는 더 쉽게 할 수도 있지만, 가독성을 높이기 위해 이와 같이 진행하였다.
            연산속도에는 큰 차이가 없어 보인다.)
        """
        right_shift = np.pad(cond[:-1], (1, 0), mode="constant", constant_values=0)
        left_shift = np.pad(cond[1:], (0, 1), mode="constant", constant_values=0)
        is_start = np.where((cond ^ right_shift) == True)[0][::2]
        is_end = np.where((cond ^ left_shift) == True)[0][1::2]
        raw_event = np.array(list(zip(is_start, is_end)))
        if len(raw_event) != 0:
            raw_event[:, 1] = raw_event[:, 1] + 1
        return raw_event

    def label_neg_event(self, arr, pos_event, config):
        """
        * 1. MAP > 75 인 데이터포인트를 전체 신호에서 찾는다.
        * 2. 단, 저혈압 이벤트의 시작지점으로부터는 40분, 종료지점으로부터는 20분 이격되어야 한다.
        """
        is_normo = arr > 75
        is_hypo_influence = self.mark_influence(arr, pos_event.copy(), config)
        raw_neg_event = self.get_segment(is_normo & ~is_hypo_influence)
        if len(raw_neg_event) == 0:
            return np.array([], dtype=np.int32)
        is_at_least_input_len = np.where(
            np.diff(raw_neg_event, axis=1) >= (config["fs"] * config["input_len"])
        )[0]
        return raw_neg_event[is_at_least_input_len]


class HypophetTest2(Hypophet):
    """
    * 저혈압 정의: 1분 동안 AUC(MAP) < 65, group_range 0.
    * 비저혈압 정의:
    * 1. MAP > 75 인 데이터포인트를 전체 신호에서 찾는다.
    * 2. 단, 저혈압 이벤트의 시작지점으로부터는 40분, 종료지점으로부터는 20분 이격되어야 한다.
    *
    * 저혈압 샘플링: 15분전부터 onset 까지의 데이터
    * 비저혈압 샘플링: 전체 * 20%
    * 못 씀.
    """

    name = "hypophettest2"

    def find_pos_onset(self, arr, config):
        input_size = config["fs"] * config["input_len"]
        win_size = config["fs"] * config["pred_len"]
        if len(arr) < win_size:
            return np.array([], dtype=np.int32)
        windows_idx = sliding_window_view(range(len(arr)), win_size)[::input_size]
        windows = sliding_window_view(arr, win_size)[::input_size]
        hypo_percentage = np.sum(windows, axis=1, keepdims=True) / windows.shape[1]
        hypo_onset = windows_idx[np.where(hypo_percentage >= 0.9)[0]]
        pos_event = hypo_onset[:, [0, -1]]
        pos_event[:, -1] = pos_event[:, -1] + 1
        return pos_event

    def mark_influence(self, arr, event, config):
        is_influence = np.zeros(len(arr)).astype(bool)
        if len(event) == 0:
            return is_influence
        event[:, 0] = event[:, 0] - config["fs"] * 60 * 40
        event[:, 1] = event[:, 1] + config["fs"] * 60 * 20
        event = np.where(event < 0, 0, event)
        event = np.where(event >= len(arr), len(arr), event)
        influence_idx = np.unique(np.hstack([np.arange(s, e) for s, e in event]))
        is_influence[influence_idx] = True
        return is_influence

    def get_segment(self, cond) -> npt.NDArray:
        """
        Input:  MAP 배열
        Return: 입력으로부터 추출한 모든 영역의 [start, end]
        Description:
            조건에 부합하는 지점들을 찾는다.
            right_shift 는 현재 기준 직전 MAP 값, left_shift 는 현재 기준 직후 MAP 값 이다.
            is_pos_start 는 추출된 모든 영역의 start idx 를 갖는다.
            is_pos_end 는 추출된 모든 영역의 end idx 를 갖는다.
            항상 start 는 start idx, end 는 end idx 를 갖게 하기 위해 constant_values 를
            right_shift, left_shift 에서 모두 0으로 두었다.
            이러면 `np.where((is_hypo ^ right_shift) == True)[0]` 은 항상 start 로 시작하고,
            마지막은 항상 end 로 끝날 것이다.
            각 영역의 [start, end] 를 넘파이 배열로 구성한다.
            `np.arange` 에 바로 입력하기 용이하도록 end 에 +1 을 해준다.
            (사실 코딩만 따졌을 때는 더 쉽게 할 수도 있지만, 가독성을 높이기 위해 이와 같이 진행하였다.
            연산속도에는 큰 차이가 없어 보인다.)
        """
        right_shift = np.pad(cond[:-1], (1, 0), mode="constant", constant_values=0)
        left_shift = np.pad(cond[1:], (0, 1), mode="constant", constant_values=0)
        is_start = np.where((cond ^ right_shift) == True)[0][::2]
        is_end = np.where((cond ^ left_shift) == True)[0][1::2]
        raw_event = np.array(list(zip(is_start, is_end)))
        if len(raw_event) != 0:
            raw_event[:, 1] = raw_event[:, 1] + 1
        return raw_event

    def label_pos_event(self, arr, config) -> npt.NDArray:
        """
        * Moghadam은 15분 확장을 샘플링 때 수행한다. 그러므로 여기서 15분 확장은 할 필요없다.
        * 그리고 nanmean을 사용했다. 사용했는데, 오히려 라벨링 퀄리티가 떨어질까봐 다시 배제하였다.
        * 예를 들어, 90%가 np.nan이더라도 나머지 10%가 < 65 미만이면 낮은 퀄리티이기 때문이다.
        """
        fs = config["fs"]
        input_len, pred_len = config["input_len"] * fs, config["pred_len"] * fs
        win_size = config["event_condition"]["pos"]["N"] * fs
        windows, win_rng = self.set_window_of(arr, win_size)
        is_window_valid = np.all(windows != 0, axis=1)
        is_positive = windows.mean(axis=1) < 65
        try:
            pos_event = win_rng[is_window_valid & is_positive]
            # pos_event = self.extend_to_past(pos_event, input_len + pred_len)
            pos_event = self.group_event(pos_event, config)
        except Exception as e:
            pos_event = np.array([], dtype=np.int32)

        return pos_event

    def label_neg_event(self, arr, pos_event, config):
        """
        * 1. MAP > 75 인 데이터포인트를 전체 신호에서 찾는다.
        * 2. 단, 저혈압 이벤트의 시작지점으로부터는 40분, 종료지점으로부터는 20분 이격되어야 한다.
        """
        is_normo = arr > 75
        is_hypo_influence = self.mark_influence(arr, pos_event.copy(), config)
        raw_neg_event = self.get_segment(is_normo & ~is_hypo_influence)
        if len(raw_neg_event) == 0:
            return np.array([], dtype=np.int32)
        is_at_least_input_len = np.where(
            np.diff(raw_neg_event, axis=1) >= (config["fs"] * config["input_len"])
        )[0]
        return raw_neg_event[is_at_least_input_len]

    def extract_pos_segment(self, events, config):
        if len(events) == 0:
            return np.array([], dtype=np.int32)
        input_size = config["fs"] * config["input_len"]
        win_size = config["fs"] * config["pred_len"]
        event_idx = np.vstack([np.arange(start, end) for start, end in events])
        pos_segment = np.vstack(
            sliding_window_view(event_idx - win_size, input_size, axis=1)[
                :, ::input_size, :
            ]
        )
        pos_segment = np.unique(pos_segment, axis=0)
        return pos_segment[:, [0, -1]]

    def extract_neg_segment(self, events, config):
        if len(events) == 0:
            return np.array([], dtype=np.int32)
        input_size = config["fs"] * config["input_len"]
        neg_segment = []
        for e in [np.arange(start, end) for start, end in events]:
            neg_segment.append(sliding_window_view(e, input_size)[::input_size])
        neg_segment = np.vstack(neg_segment)
        neg_segment = neg_segment[:, [0, -1]]
        rand_idx = np.random.choice(
            range(len(neg_segment)), round(len(neg_segment) * 0.2)
        )
        return neg_segment[rand_idx]

    def extract_segment(self, events, config, label):
        np.random.seed(config["seed"])
        if label == "pos":
            seg_range = self.extract_pos_segment(events.copy(), config)
        elif label == "neg":
            seg_range = self.extract_neg_segment(events.copy(), config)
        else:
            print("Not implemented.")
        return seg_range


class HypophetTest3(Hypophet):
    """
    * 샘플링 방식만 바꿔보았다.
    * 라벨링은 Hypophet 스타일:
    *   - AUC(MAP) < 65, group_range 5.
    *
    * 샘플링은 아래와 같이 변형하였음:
    *   - 양성 이벤트는 앞쪽 절반만 사용. 뒤쪽 절반은 이미 의료개입이 일어났을 수 있기 때문이다.
    *   - 음성은 샘플링 중 20%만 활용 (Moghadam 방식)
    * 못 씀.
    """

    name = "hypophettest3"

    def extract_pos_segment(self, events, config):
        if len(events) == 0:
            return np.array([], dtype=np.int32)
        input_size = config["fs"] * config["input_len"]
        win_size = config["fs"] * config["pred_len"]

        # 양성 이벤트의 앞쪽 절반만 가져온다.
        event_idx = [
            np.arange(start, start + ((end - start) // 2)) for start, end in events
        ]
        # 앞쪽 절반을 input size 씩 이동하면서 가져온다.
        pos_segment = np.vstack(
            [sliding_window_view(e, input_size)[::input_size] for e in event_idx]
        )
        pos_segment = np.unique(pos_segment, axis=0)

        return pos_segment[:, [0, -1]]

    def extract_neg_segment(self, events, config):
        """
        Extract neg를 input size 간격으로 하고 20%만 취하는 방식을 할까?
        """
        if len(events) == 0:
            return np.array([], dtype=np.int32)
        input_size = config["fs"] * config["input_len"]
        neg_segment = []
        for e in [np.arange(start, end) for start, end in events]:
            neg_segment.append(sliding_window_view(e, input_size)[::input_size])
        neg_segment = np.vstack(neg_segment)
        neg_segment = neg_segment[:, [0, -1]]
        rand_idx = np.random.choice(
            range(len(neg_segment)), round(len(neg_segment) * 0.2)
        )
        return neg_segment[rand_idx]

    def extract_segment(self, events, config, label):
        np.random.seed(config["seed"])
        if label == "pos":
            seg_range = self.extract_pos_segment(events.copy(), config)
        elif label == "neg":
            seg_range = self.extract_neg_segment(events.copy(), config)
        else:
            print("Not implemented.")
        return seg_range


class HypophetTest4(Hypophet):
    """
    * 보다 엄밀하게 양성을 정의해보자.
    * 저혈압 정의: 15분 동안 AUC(MAP) < 65, group_range 0.
    * 비저혈압 정의:
    * 1. MAP > 75 인 데이터포인트를 전체 신호에서 찾는다.
    * 2. 단, 저혈압 이벤트의 시작지점으로부터는 40분, 종료지점으로부터는 20분 이격되어야 한다.
    *
    * 저혈압 샘플링: 15분전부터 onset 까지의 데이터
    * 비저혈압 샘플링: 전체 * 20%
    *
    * AUC(MAP) < 65 인 것 빼고는 Moghadam 스타일.
    * 못 씀.
    """

    name = "hypophettest4"

    def mark_influence(self, arr, event, config):
        is_influence = np.zeros(len(arr)).astype(bool)
        if len(event) == 0:
            return is_influence
        event[:, 0] = event[:, 0] - config["fs"] * 60 * 40
        event[:, 1] = event[:, 1] + config["fs"] * 60 * 20
        event = np.where(event < 0, 0, event)
        event = np.where(event >= len(arr), len(arr), event)
        influence_idx = np.unique(np.hstack([np.arange(s, e) for s, e in event]))
        is_influence[influence_idx] = True
        return is_influence

    def get_segment(self, cond) -> npt.NDArray:
        """
        Input:  MAP 배열
        Return: 입력으로부터 추출한 모든 영역의 [start, end]
        Description:
            조건에 부합하는 지점들을 찾는다.
            right_shift 는 현재 기준 직전 MAP 값, left_shift 는 현재 기준 직후 MAP 값 이다.
            is_pos_start 는 추출된 모든 영역의 start idx 를 갖는다.
            is_pos_end 는 추출된 모든 영역의 end idx 를 갖는다.
            항상 start 는 start idx, end 는 end idx 를 갖게 하기 위해 constant_values 를
            right_shift, left_shift 에서 모두 0으로 두었다.
            이러면 `np.where((is_hypo ^ right_shift) == True)[0]` 은 항상 start 로 시작하고,
            마지막은 항상 end 로 끝날 것이다.
            각 영역의 [start, end] 를 넘파이 배열로 구성한다.
            `np.arange` 에 바로 입력하기 용이하도록 end 에 +1 을 해준다.
            (사실 코딩만 따졌을 때는 더 쉽게 할 수도 있지만, 가독성을 높이기 위해 이와 같이 진행하였다.
            연산속도에는 큰 차이가 없어 보인다.)
        """
        right_shift = np.pad(cond[:-1], (1, 0), mode="constant", constant_values=0)
        left_shift = np.pad(cond[1:], (0, 1), mode="constant", constant_values=0)
        is_start = np.where((cond ^ right_shift) == True)[0][::2]
        is_end = np.where((cond ^ left_shift) == True)[0][1::2]
        raw_event = np.array(list(zip(is_start, is_end)))
        if len(raw_event) != 0:
            raw_event[:, 1] = raw_event[:, 1] + 1
        return raw_event

    def label_pos_event(self, arr, config):
        fs = config["fs"]
        input_len, pred_len = config["input_len"] * fs, config["pred_len"] * fs
        win_size = config["event_condition"]["pos"]["N"] * fs * 15  # 15 min
        windows, win_rng = self.set_window_of(arr, win_size)
        if len(windows) == 0:
            return np.array([], dtype=np.int32)
        # is_window_valid = np.all(windows != 0, axis=1)
        is_positive = windows.mean(axis=1) < 65

        try:
            pos_event = win_rng[is_positive]
            # pos_event = self.extend_to_past(pos_event, input_len + pred_len)
            pos_event = self.group_event(pos_event, config)
        except Exception as e:
            pos_event = np.array([], dtype=np.int32)

        return pos_event

    def label_neg_event(self, arr, pos_event, config):
        is_normo = arr > 75
        is_hypo_influence = self.mark_influence(arr, pos_event.copy(), config)
        raw_neg_event = self.get_segment(is_normo & ~is_hypo_influence)
        if len(raw_neg_event) == 0:
            return np.array([], dtype=np.int32)
        is_at_least_input_len = np.where(
            np.diff(raw_neg_event, axis=1) >= (config["fs"] * config["input_len"])
        )[0]
        return raw_neg_event[is_at_least_input_len]

    def extract_pos_segment(self, events, config):
        if len(events) == 0:
            return np.array([], dtype=np.int32)
        input_size = config["fs"] * config["input_len"]
        win_size = config["fs"] * config["pred_len"]
        event_idx = np.vstack([np.arange(start, end) for start, end in events])
        pos_segment = np.vstack(
            sliding_window_view(event_idx - win_size, input_size, axis=1)[
                :, ::input_size, :
            ]
        )
        pos_segment = np.unique(pos_segment, axis=0)
        return pos_segment[:, [0, -1]]

    def extract_neg_segment(self, events, config):
        if len(events) == 0:
            return np.array([], dtype=np.int32)
        input_size = config["fs"] * config["input_len"]
        neg_segment = []
        for e in [np.arange(start, end) for start, end in events]:
            neg_segment.append(sliding_window_view(e, input_size)[::input_size])
        neg_segment = np.vstack(neg_segment)
        neg_segment = neg_segment[:, [0, -1]]
        rand_idx = np.random.choice(
            range(len(neg_segment)), round(len(neg_segment) * 0.2)
        )
        return neg_segment[rand_idx]

    def extract_segment(self, events, config, label):
        np.random.seed(config["seed"])
        if label == "pos":
            seg_range = self.extract_pos_segment(events.copy(), config)
        elif label == "neg":
            seg_range = self.extract_neg_segment(events.copy(), config)
        else:
            print("Not implemented.")
        return seg_range


class HypophetTest5(Hypophet):
    """
    * Time-Weighted Average(TWA) MAP < 65를 사용해보면 어떨까?
    * unittest_label_pos2.ipynb 참고.
    * 못 씀.
    """

    name = "hypophettest5"

    def calc_twa(self, windows):
        if len(windows) == 0:
            return np.array([], dtype=np.float32)
        deficit = 65 - np.where(windows == 0, np.nan, windows).round()
        twa = np.nanmean(np.where(deficit < 0, 0, deficit), axis=1)
        return twa

    def mark_influence(self, arr, event, config):
        is_influence = np.zeros(len(arr)).astype(bool)
        if len(event) == 0:
            return is_influence
        event[:, 0] = event[:, 0] - config["fs"] * 60 * 40
        event[:, 1] = event[:, 1] + config["fs"] * 60 * 20
        event = np.where(event < 0, 0, event)
        event = np.where(event >= len(arr), len(arr), event)
        influence_idx = np.unique(np.hstack([np.arange(s, e) for s, e in event]))
        is_influence[influence_idx] = True
        return is_influence

    def get_segment(self, cond) -> npt.NDArray:
        """
        Input:  MAP 배열
        Return: 입력으로부터 추출한 모든 영역의 [start, end]
        Description:
            조건에 부합하는 지점들을 찾는다.
            right_shift 는 현재 기준 직전 MAP 값, left_shift 는 현재 기준 직후 MAP 값 이다.
            is_pos_start 는 추출된 모든 영역의 start idx 를 갖는다.
            is_pos_end 는 추출된 모든 영역의 end idx 를 갖는다.
            항상 start 는 start idx, end 는 end idx 를 갖게 하기 위해 constant_values 를
            right_shift, left_shift 에서 모두 0으로 두었다.
            이러면 `np.where((is_hypo ^ right_shift) == True)[0]` 은 항상 start 로 시작하고,
            마지막은 항상 end 로 끝날 것이다.
            각 영역의 [start, end] 를 넘파이 배열로 구성한다.
            `np.arange` 에 바로 입력하기 용이하도록 end 에 +1 을 해준다.
            (사실 코딩만 따졌을 때는 더 쉽게 할 수도 있지만, 가독성을 높이기 위해 이와 같이 진행하였다.
            연산속도에는 큰 차이가 없어 보인다.)
        """
        right_shift = np.pad(cond[:-1], (1, 0), mode="constant", constant_values=0)
        left_shift = np.pad(cond[1:], (0, 1), mode="constant", constant_values=0)
        is_start = np.where((cond ^ right_shift) == True)[0][::2]
        is_end = np.where((cond ^ left_shift) == True)[0][1::2]
        raw_event = np.array(list(zip(is_start, is_end)))
        if len(raw_event) != 0:
            raw_event[:, 1] = raw_event[:, 1] + 1
        return raw_event

    def label_pos_event(self, arr, config):
        fs = config["fs"]
        # input_len, pred_len = config["input_len"] * fs, config["pred_len"] * fs
        win_size = config["event_condition"]["pos"]["N"] * fs * 15  # 15 min
        windows, win_rng = self.set_window_of(arr, win_size)
        if len(windows) == 0:
            return np.array([], dtype=np.int32)
        # is_window_valid = np.all(windows != 0, axis=1)
        is_positive = self.calc_twa(windows) >= 0.19
        try:
            pos_event = win_rng[is_positive]
            # pos_event = self.extend_to_past(pos_event, input_len + pred_len)
            pos_event = self.group_event(pos_event, config)
        except Exception as e:
            pos_event = np.array([], dtype=np.int32)
        return pos_event

    def label_neg_event(self, arr, pos_event, config):
        is_normo = arr > 75
        is_hypo_influence = self.mark_influence(arr, pos_event.copy(), config)
        raw_neg_event = self.get_segment(is_normo & ~is_hypo_influence)
        if len(raw_neg_event) == 0:
            return np.array([], dtype=np.int32)
        is_at_least_input_len = np.where(
            np.diff(raw_neg_event, axis=1) >= (config["fs"] * config["input_len"])
        )[0]
        return raw_neg_event[is_at_least_input_len]

    def extract_pos_segment(self, events, config):
        if len(events) == 0:
            return np.array([], dtype=np.int32)
        input_size = config["fs"] * config["input_len"]
        win_size = config["fs"] * config["pred_len"]
        event_idx = np.vstack([np.arange(start, end) for start, end in events])
        pos_segment = np.vstack(
            sliding_window_view(event_idx - win_size, input_size, axis=1)[
                :, ::input_size, :
            ]
        )
        pos_segment = np.unique(pos_segment, axis=0)
        return pos_segment[:, [0, -1]]

    def extract_neg_segment(self, events, config):
        if len(events) == 0:
            return np.array([], dtype=np.int32)
        input_size = config["fs"] * config["input_len"]
        neg_segment = []
        for e in [np.arange(start, end) for start, end in events]:
            neg_segment.append(sliding_window_view(e, input_size)[::input_size])
        neg_segment = np.vstack(neg_segment)
        neg_segment = neg_segment[:, [0, -1]]
        rand_idx = np.random.choice(
            range(len(neg_segment)), round(len(neg_segment) * 0.2)
        )
        return neg_segment[rand_idx]

    def extract_segment(self, events, config, label):
        np.random.seed(config["seed"])
        if label == "pos":
            seg_range = self.extract_pos_segment(events.copy(), config)
        elif label == "neg":
            seg_range = self.extract_neg_segment(events.copy(), config)
        else:
            print("Not implemented.")
        return seg_range


class HypophetTest6(Hypophet):
    """
    * Moghadam(15분) 과 TWA-MAP(15분) 둘 다 만족하는 경우
    * 가능성 있음.
    """

    name = "hypophettest6"

    def calc_twa(self, windows):
        if len(windows) == 0:
            return np.array([], dtype=np.float32)
        deficit = 65 - np.where(windows == 0, np.nan, windows).round()
        twa = np.nanmean(np.where(deficit < 0, 0, deficit), axis=1)
        return twa

    def find_pos_onset(self, arr, config):
        input_size = config["fs"] * config["input_len"]
        win_size = config["fs"] * config["pred_len"]
        if len(arr) < win_size:
            return np.array([], dtype=np.int32)
        windows_idx = sliding_window_view(range(len(arr) + 1), win_size)[::input_size]
        windows = sliding_window_view(arr, win_size)[::input_size]
        hypo_percentage = np.sum(windows, axis=1, keepdims=True) / windows.shape[1]
        hypo_onset = windows_idx[np.where(hypo_percentage >= 0.9)[0]]
        pos_event = hypo_onset[:, [0, -1]]
        pos_event[:, -1] = pos_event[:, -1] + 1
        return pos_event

    def mark_influence(self, arr, event, config):
        is_influence = np.zeros(len(arr)).astype(bool)
        if len(event) == 0:
            return is_influence
        event[:, 0] = event[:, 0] - config["fs"] * 60 * 40
        event[:, 1] = event[:, 1] + config["fs"] * 60 * 20
        event = np.where(event < 0, 0, event)
        event = np.where(event >= len(arr), len(arr), event)
        influence_idx = np.unique(np.hstack([np.arange(s, e) for s, e in event]))
        is_influence[influence_idx] = True
        return is_influence

    def get_segment(self, cond) -> npt.NDArray:
        """
        Input:  MAP 배열
        Return: 입력으로부터 추출한 모든 영역의 [start, end]
        Description:
            조건에 부합하는 지점들을 찾는다.
            right_shift 는 현재 기준 직전 MAP 값, left_shift 는 현재 기준 직후 MAP 값 이다.
            is_pos_start 는 추출된 모든 영역의 start idx 를 갖는다.
            is_pos_end 는 추출된 모든 영역의 end idx 를 갖는다.
            항상 start 는 start idx, end 는 end idx 를 갖게 하기 위해 constant_values 를
            right_shift, left_shift 에서 모두 0으로 두었다.
            이러면 `np.where((is_hypo ^ right_shift) == True)[0]` 은 항상 start 로 시작하고,
            마지막은 항상 end 로 끝날 것이다.
            각 영역의 [start, end] 를 넘파이 배열로 구성한다.
            `np.arange` 에 바로 입력하기 용이하도록 end 에 +1 을 해준다.
            (사실 코딩만 따졌을 때는 더 쉽게 할 수도 있지만, 가독성을 높이기 위해 이와 같이 진행하였다.
            연산속도에는 큰 차이가 없어 보인다.)
        """
        right_shift = np.pad(cond[:-1], (1, 0), mode="constant", constant_values=0)
        left_shift = np.pad(cond[1:], (0, 1), mode="constant", constant_values=0)
        is_start = np.where((cond ^ right_shift) == True)[0][::2]
        is_end = np.where((cond ^ left_shift) == True)[0][1::2]
        raw_event = np.array(list(zip(is_start, is_end)))
        if len(raw_event) != 0:
            raw_event[:, 1] = raw_event[:, 1] + 1
        return raw_event

    def label_pos_event(self, arr, config):
        """
        * 1. MAP < 65 인 데이터포인트를 전체 신호에서 찾는다.
        * 2. 위 조건이 True 인 경우가 15분 동안 90% 이상인 경우를 저혈압 이벤트로 추출한다.
        """
        event_map = np.zeros(len(arr), dtype=bool)
        try:
            is_pos_orig = self.find_pos_onset((arr < 65) & (arr > 0), config)
            pos_orig_event = self.group_event(is_pos_orig, config)
            pos_orig_map = event_map.copy()
            pos_orig_map[np.hstack([range(*e) for e in pos_orig_event])] = True
        except:
            pos_orig_map = event_map.copy()

        input_size = config["fs"] * config["input_len"]
        win_size = config["fs"] * config["pred_len"]
        if len(arr) < win_size:
            return np.array([], dtype=np.int32)
        windows_idx = sliding_window_view(range(len(arr)), win_size)[::input_size]
        windows = sliding_window_view(arr, win_size)[::input_size]
        if len(windows) == 0:
            return np.array([], dtype=np.int32)

        try:
            is_pos_twa = self.calc_twa(windows) >= 0.19
            pos_twa_event = self.group_event(windows_idx[is_pos_twa], config)
            pos_twa_map = event_map.copy()
            pos_twa_map[np.hstack([range(*e) for e in pos_twa_event])] = True
        except:
            pos_twa_map = event_map.copy()

        pos_event = self.get_segment(pos_orig_map & pos_twa_map)
        return pos_event

    def label_neg_event(self, arr, pos_event, config):
        """
        * 1. MAP > 75 인 데이터포인트를 전체 신호에서 찾는다.
        * 2. 단, 저혈압 이벤트의 시작지점으로부터는 40분, 종료지점으로부터는 20분 이격되어야 한다.
        """
        is_normo = arr > 75
        is_hypo_influence = self.mark_influence(arr, pos_event.copy(), config)
        raw_neg_event = self.get_segment(is_normo & ~is_hypo_influence)
        if len(raw_neg_event) == 0:
            return np.array([], dtype=np.int32)
        is_at_least_input_len = np.where(
            np.diff(raw_neg_event, axis=1) >= (config["fs"] * config["input_len"])
        )[0]
        return raw_neg_event[is_at_least_input_len]

    def extract_pos_segment(self, events, config):
        if len(events) == 0:
            return np.array([], dtype=np.int32)
        input_size = config["fs"] * config["input_len"]
        win_size = config["fs"] * config["pred_len"]
        event_idx = [np.arange(start, end) for start, end in events]

        pos_segment = np.vstack(
            [
                sliding_window_view(e - win_size, input_size)[::input_size]
                for e in event_idx
            ]
        )
        return pos_segment[:, [0, -1]]

    def extract_neg_segment(self, events, config):
        if len(events) == 0:
            return np.array([], dtype=np.int32)
        input_size = config["fs"] * config["input_len"]
        neg_segment = []
        for e in [np.arange(start, end) for start, end in events]:
            neg_segment.append(sliding_window_view(e, input_size)[::input_size])
        neg_segment = np.vstack(neg_segment)
        neg_segment = neg_segment[:, [0, -1]]
        rand_idx = np.random.choice(
            range(len(neg_segment)), round(len(neg_segment) * 0.2)
        )
        return neg_segment[rand_idx]

    def extract_segment(self, events, config, label):
        np.random.seed(config["seed"])
        if label == "pos":
            seg_range = self.extract_pos_segment(events.copy(), config)
        elif label == "neg":
            seg_range = self.extract_neg_segment(events.copy(), config)
        else:
            print("Not implemented.")
        return seg_range


class HypophetTest7(Hypophet):
    """
    * Moghadam(15분) 과 TWA-MAP(15분) 중 하나만 만족하는 경우
    * 못 씀.
    """

    name = "hypophettest7"

    def calc_twa(self, windows):
        if len(windows) == 0:
            return np.array([], dtype=np.float32)
        deficit = 65 - np.where(windows == 0, np.nan, windows).round()
        twa = np.nanmean(np.where(deficit < 0, 0, deficit), axis=1)
        return twa

    def find_pos_onset(self, arr, config):
        input_size = config["fs"] * config["input_len"]
        win_size = config["fs"] * config["pred_len"]
        if len(arr) < win_size:
            return np.array([], dtype=np.int32)
        windows_idx = sliding_window_view(range(len(arr) + 1), win_size)[::input_size]
        windows = sliding_window_view(arr, win_size)[::input_size]
        hypo_percentage = np.sum(windows, axis=1, keepdims=True) / windows.shape[1]
        hypo_onset = windows_idx[np.where(hypo_percentage >= 0.9)[0]]
        pos_event = hypo_onset[:, [0, -1]]
        pos_event[:, -1] = pos_event[:, -1] + 1
        return pos_event

    def mark_influence(self, arr, event, config):
        is_influence = np.zeros(len(arr)).astype(bool)
        if len(event) == 0:
            return is_influence
        event[:, 0] = event[:, 0] - config["fs"] * 60 * 40
        event[:, 1] = event[:, 1] + config["fs"] * 60 * 20
        event = np.where(event < 0, 0, event)
        event = np.where(event >= len(arr), len(arr), event)
        influence_idx = np.unique(np.hstack([np.arange(s, e) for s, e in event]))
        is_influence[influence_idx] = True
        return is_influence

    def get_segment(self, cond) -> npt.NDArray:
        """
        Input:  MAP 배열
        Return: 입력으로부터 추출한 모든 영역의 [start, end]
        Description:
            조건에 부합하는 지점들을 찾는다.
            right_shift 는 현재 기준 직전 MAP 값, left_shift 는 현재 기준 직후 MAP 값 이다.
            is_pos_start 는 추출된 모든 영역의 start idx 를 갖는다.
            is_pos_end 는 추출된 모든 영역의 end idx 를 갖는다.
            항상 start 는 start idx, end 는 end idx 를 갖게 하기 위해 constant_values 를
            right_shift, left_shift 에서 모두 0으로 두었다.
            이러면 `np.where((is_hypo ^ right_shift) == True)[0]` 은 항상 start 로 시작하고,
            마지막은 항상 end 로 끝날 것이다.
            각 영역의 [start, end] 를 넘파이 배열로 구성한다.
            `np.arange` 에 바로 입력하기 용이하도록 end 에 +1 을 해준다.
            (사실 코딩만 따졌을 때는 더 쉽게 할 수도 있지만, 가독성을 높이기 위해 이와 같이 진행하였다.
            연산속도에는 큰 차이가 없어 보인다.)
        """
        right_shift = np.pad(cond[:-1], (1, 0), mode="constant", constant_values=0)
        left_shift = np.pad(cond[1:], (0, 1), mode="constant", constant_values=0)
        is_start = np.where((cond ^ right_shift) == True)[0][::2]
        is_end = np.where((cond ^ left_shift) == True)[0][1::2]
        raw_event = np.array(list(zip(is_start, is_end)))
        if len(raw_event) != 0:
            raw_event[:, 1] = raw_event[:, 1] + 1
        return raw_event

    def label_pos_event(self, arr, config):
        """
        * 1. MAP < 65 인 데이터포인트를 전체 신호에서 찾는다.
        * 2. 위 조건이 True 인 경우가 15분 동안 90% 이상인 경우를 저혈압 이벤트로 추출한다.
        """
        event_map = np.zeros(len(arr), dtype=bool)
        try:
            is_pos_orig = self.find_pos_onset((arr < 65) & (arr > 0), config)
            pos_orig_event = self.group_event(is_pos_orig, config)
            pos_orig_map = event_map.copy()
            pos_orig_map[np.hstack([range(*e) for e in pos_orig_event])] = True
        except:
            pos_orig_map = event_map.copy()

        input_size = config["fs"] * config["input_len"]
        win_size = config["fs"] * config["pred_len"]
        if len(arr) < win_size:
            return np.array([], dtype=np.int32)
        windows_idx = sliding_window_view(range(len(arr)), win_size)[::input_size]
        windows = sliding_window_view(arr, win_size)[::input_size]
        if len(windows) == 0:
            return np.array([], dtype=np.int32)

        try:
            is_pos_twa = self.calc_twa(windows) >= 0.19
            pos_twa_event = self.group_event(windows_idx[is_pos_twa], config)
            pos_twa_map = event_map.copy()
            pos_twa_map[np.hstack([range(*e) for e in pos_twa_event])] = True
        except:
            pos_twa_map = event_map.copy()

        pos_event = self.get_segment(pos_orig_map | pos_twa_map)
        return pos_event

    def label_neg_event(self, arr, pos_event, config):
        """
        * 1. MAP > 75 인 데이터포인트를 전체 신호에서 찾는다.
        * 2. 단, 저혈압 이벤트의 시작지점으로부터는 40분, 종료지점으로부터는 20분 이격되어야 한다.
        """
        is_normo = arr > 75
        is_hypo_influence = self.mark_influence(arr, pos_event.copy(), config)
        raw_neg_event = self.get_segment(is_normo & ~is_hypo_influence)
        if len(raw_neg_event) == 0:
            return np.array([], dtype=np.int32)
        is_at_least_input_len = np.where(
            np.diff(raw_neg_event, axis=1) >= (config["fs"] * config["input_len"])
        )[0]
        return raw_neg_event[is_at_least_input_len]

    def extract_pos_segment(self, events, config):
        if len(events) == 0:
            return np.array([], dtype=np.int32)
        input_size = config["fs"] * config["input_len"]
        win_size = config["fs"] * config["pred_len"]
        event_idx = [np.arange(start, end) for start, end in events]

        pos_segment = np.vstack(
            [
                sliding_window_view(e - win_size, input_size)[::input_size]
                for e in event_idx
            ]
        )
        return pos_segment[:, [0, -1]]

    def extract_neg_segment(self, events, config):
        if len(events) == 0:
            return np.array([], dtype=np.int32)
        input_size = config["fs"] * config["input_len"]
        neg_segment = []
        for e in [np.arange(start, end) for start, end in events]:
            neg_segment.append(sliding_window_view(e, input_size)[::input_size])
        neg_segment = np.vstack(neg_segment)
        neg_segment = neg_segment[:, [0, -1]]
        rand_idx = np.random.choice(
            range(len(neg_segment)), round(len(neg_segment) * 0.2)
        )
        return neg_segment[rand_idx]

    def extract_segment(self, events, config, label):
        np.random.seed(config["seed"])
        if label == "pos":
            seg_range = self.extract_pos_segment(events.copy(), config)
        elif label == "neg":
            seg_range = self.extract_neg_segment(events.copy(), config)
        else:
            print("Not implemented.")
        return seg_range


class HypophetTest8(Hypophet):
    """
    Moghadam(15분) 과 TWA-MAP(각 moghadam 이벤트의 시작점부터 끝까지) 둘 다 만족하는 경우
    Moghadam을 메인, TWA-MAP를 서브로 사용하는 방식.
    """

    name = "hypophettest8"

    def calc_twa(self, windows):
        if len(windows) == 0:
            return np.array([], dtype=np.float32)
        deficit = 65 - np.where(windows == 0, np.nan, windows).round()
        twa = np.nanmean(np.where(deficit < 0, 0, deficit), axis=-1)
        return twa

    def find_pos_onset(self, arr, config):
        input_size = config["fs"] * config["input_len"]
        win_size = config["fs"] * config["pred_len"]
        if len(arr) < win_size:
            return np.array([], dtype=np.int32)
        windows_idx = sliding_window_view(range(len(arr) + 1), win_size)[::input_size]
        windows = sliding_window_view(arr, win_size)[::input_size]
        hypo_percentage = np.sum(windows, axis=1, keepdims=True) / windows.shape[1]
        hypo_onset = windows_idx[np.where(hypo_percentage >= 0.9)[0]]
        pos_event = hypo_onset[:, [0, -1]]
        pos_event[:, -1] = pos_event[:, -1] + 1
        return pos_event

    def mark_influence(self, arr, event, config):
        is_influence = np.zeros(len(arr)).astype(bool)
        if len(event) == 0:
            return is_influence
        event[:, 0] = event[:, 0] - config["fs"] * 60 * 40
        event[:, 1] = event[:, 1] + config["fs"] * 60 * 20
        event = np.where(event < 0, 0, event)
        event = np.where(event >= len(arr), len(arr), event)
        influence_idx = np.unique(np.hstack([np.arange(s, e) for s, e in event]))
        is_influence[influence_idx] = True
        return is_influence

    def get_segment(self, cond) -> npt.NDArray:
        """
        Input:  MAP 배열
        Return: 입력으로부터 추출한 모든 영역의 [start, end]
        Description:
            조건에 부합하는 지점들을 찾는다.
            right_shift 는 현재 기준 직전 MAP 값, left_shift 는 현재 기준 직후 MAP 값 이다.
            is_pos_start 는 추출된 모든 영역의 start idx 를 갖는다.
            is_pos_end 는 추출된 모든 영역의 end idx 를 갖는다.
            항상 start 는 start idx, end 는 end idx 를 갖게 하기 위해 constant_values 를
            right_shift, left_shift 에서 모두 0으로 두었다.
            이러면 `np.where((is_hypo ^ right_shift) == True)[0]` 은 항상 start 로 시작하고,
            마지막은 항상 end 로 끝날 것이다.
            각 영역의 [start, end] 를 넘파이 배열로 구성한다.
            `np.arange` 에 바로 입력하기 용이하도록 end 에 +1 을 해준다.
            (사실 코딩만 따졌을 때는 더 쉽게 할 수도 있지만, 가독성을 높이기 위해 이와 같이 진행하였다.
            연산속도에는 큰 차이가 없어 보인다.)
        """
        right_shift = np.pad(cond[:-1], (1, 0), mode="constant", constant_values=0)
        left_shift = np.pad(cond[1:], (0, 1), mode="constant", constant_values=0)
        is_start = np.where((cond ^ right_shift) == True)[0][::2]
        is_end = np.where((cond ^ left_shift) == True)[0][1::2]
        raw_event = np.array(list(zip(is_start, is_end)))
        if len(raw_event) != 0:
            raw_event[:, 1] = raw_event[:, 1] + 1
        return raw_event

    def label_pos_event(self, arr, config):
        """
        * 1. MAP < 65 인 데이터포인트를 전체 신호에서 찾는다.
        * 2. 위 조건이 True 인 경우가 15분 동안 90% 이상인 경우를 저혈압 이벤트로 추출한다.
        """
        pos_orig_event = self.find_pos_onset((arr < 65) & (arr > 0), config)
        if len(pos_orig_event) == 0:
            return np.array([], dtype=np.float32)
        pos_orig_event = self.group_event(pos_orig_event, config)
        pos_twa_event = np.array(
            [self.calc_twa(arr[es:]) for es in pos_orig_event[:, 0]]
        )
        pos_event = pos_orig_event[pos_twa_event >= 4.7]
        return pos_event

    def label_neg_event(self, arr, pos_event, config):
        """
        * 1. MAP > 75 인 데이터포인트를 전체 신호에서 찾는다.
        * 2. 단, 저혈압 이벤트의 시작지점으로부터는 40분, 종료지점으로부터는 20분 이격되어야 한다.
        """
        is_normo = arr > 75
        is_hypo_influence = self.mark_influence(arr, pos_event.copy(), config)
        raw_neg_event = self.get_segment(is_normo & ~is_hypo_influence)
        if len(raw_neg_event) == 0:
            return np.array([], dtype=np.int32)
        is_at_least_input_len = np.where(
            np.diff(raw_neg_event, axis=1) >= (config["fs"] * config["input_len"])
        )[0]
        return raw_neg_event[is_at_least_input_len]

    def extract_pos_segment(self, events, config):
        if len(events) == 0:
            return np.array([], dtype=np.int32)
        input_size = config["fs"] * config["input_len"]
        win_size = config["fs"] * config["pred_len"]
        event_idx = [np.arange(start, end) for start, end in events]

        pos_segment = np.vstack(
            [
                sliding_window_view(e - win_size, input_size)[::input_size]
                for e in event_idx
            ]
        )
        return pos_segment[:, [0, -1]]

    def extract_neg_segment(self, events, config):
        if len(events) == 0:
            return np.array([], dtype=np.int32)
        input_size = config["fs"] * config["input_len"]
        neg_segment = []
        for e in [np.arange(start, end) for start, end in events]:
            neg_segment.append(sliding_window_view(e, input_size)[::input_size])
        neg_segment = np.vstack(neg_segment)
        neg_segment = neg_segment[:, [0, -1]]
        rand_idx = np.random.choice(
            range(len(neg_segment)), round(len(neg_segment) * 0.2)
        )
        return neg_segment[rand_idx]

    def extract_segment(self, events, config, label):
        np.random.seed(config["seed"])
        if label == "pos":
            seg_range = self.extract_pos_segment(events.copy(), config)
        elif label == "neg":
            seg_range = self.extract_neg_segment(events.copy(), config)
        else:
            print("Not implemented.")
        return seg_range
