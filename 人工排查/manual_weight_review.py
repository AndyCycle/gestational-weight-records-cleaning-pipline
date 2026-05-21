#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pregnancy weight manual review tool.

Features:
1) Visualize one sample's trajectory (gestation_day vs weight).
2) Click a point in plot/table, then apply: x2, /2, +10, -10, delete.
3) Keep source CSV read-only; persist edits into an operation log.
4) Export final corrected CSV after manual review.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tkinter as tk
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from tkinter import filedialog, messagebox, ttk

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


APP_TITLE = "妊娠期体重人工排查工具"
DEFAULT_SOURCE_REL = Path("..") / "清洗" / "06.csv"
NUMERIC_COLS = ["gestation_day", "weight", "BMI", "height", "weight_raw_p1"]
REQUIRED_COLS = ["项目流水号", "gestation_day", "weight"]
LOG_COLUMNS = [
    "timestamp",
    "operation",
    "row_id",
    "sample_id",
    "gestation_day",
    "old_weight",
    "new_weight",
    "old_bmi",
    "new_bmi",
    "old_deleted",
    "new_deleted",
    "csv_row_count",
]


def natural_key(value: str) -> Tuple[int, float, str]:
    text = str(value).strip()
    try:
        return (0, float(text), text)
    except ValueError:
        return (1, 0.0, text)


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def fmt_number(value, digits: int = 3) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


@dataclass
class EditResult:
    row_id: int
    sample_id: str
    old_weight: float
    new_weight: float
    old_bmi: float
    new_bmi: float
    old_deleted: bool
    new_deleted: bool
    operation: str


class ReviewStore:
    def __init__(self, source_path: Path, work_dir: Path):
        self.source_path = source_path
        self.work_dir = work_dir
        self.data_dir = work_dir / "data"
        self.export_dir = work_dir / "exports"
        self.log_path = self.data_dir / "edit_log.csv"

        self.df: Optional[pd.DataFrame] = None
        self.original_columns: List[str] = []
        self.sample_ids: List[str] = []
        self.sample_index: Dict[str, np.ndarray] = {}
        self.history: List[EditResult] = []
        self._update_log_path()

    def _update_log_path(self) -> None:
        stem = self.source_path.stem.strip()
        safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem)
        safe_stem = safe_stem.strip("_") or "source"
        self.log_path = self.data_dir / f"edit_log_{safe_stem}.csv"

    def set_source_path(self, source_path: Path) -> None:
        self.source_path = source_path
        self._update_log_path()

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> Dict[str, int]:
        self.ensure_dirs()
        self._update_log_path()
        if not self.source_path.exists():
            raise FileNotFoundError(f"找不到输入文件: {self.source_path}")

        df = pd.read_csv(self.source_path, low_memory=False)
        self.original_columns = list(df.columns)
        self._csv_row_count = len(df)  # 记录原始行数，用于 replay 校验

        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"缺少必要字段: {missing}")

        df["_row_id"] = np.arange(len(df), dtype=np.int64)
        df["_deleted"] = False
        df.set_index("_row_id", inplace=True, drop=False)

        df["项目流水号"] = (
            df["项目流水号"]
            .astype("string")
            .fillna("")
            .str.strip()
            .replace({"<NA>": "", "nan": ""})
        )

        for col in NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 兜底：weight 为 NaN 且来源为 Initial_Raw 时，尝试用 weight_raw_p1 填补。
        # 仅限 type='Initial_Raw'（day=0 种子行，因 Step 01 NaN bug 导致 weight 未写入）。
        # HIS 序列行（type 为空）若 weight=NaN 通常是 Step 07 主动置空的异常点，不应恢复。
        if "weight_raw_p1" in df.columns and "type" in df.columns:
            fallback_mask = (
                df["weight"].isna()
                & df["weight_raw_p1"].notna()
                & (df["type"].astype(str).str.strip() == "Initial_Raw")
            )
            if fallback_mask.any():
                df.loc[fallback_mask, "weight"] = df.loc[fallback_mask, "weight_raw_p1"]
                print(f"  [兜底] 已用 weight_raw_p1 填补 {fallback_mask.sum()} 行 Initial_Raw 种子行的 weight 缺失值")
            # 统计 HIS 行中被保护的 NaN（pipeline 主动置空，不填补）
            protected_mask = (
                df["weight"].isna()
                & df["weight_raw_p1"].notna()
                & (df["type"].astype(str).str.strip() != "Initial_Raw")
            )
            if protected_mask.any():
                print(f"  [保护] {protected_mask.sum()} 行 HIS 序列行 weight=NaN 为 pipeline 主动置空，已跳过填补")

        df["项目流水号"] = df["项目流水号"].astype("category")

        self.df = df
        self.history = []
        self.sample_ids = sorted(
            [sid for sid in df["项目流水号"].cat.categories.tolist() if sid],
            key=natural_key,
        )
        self.sample_index = df.groupby("项目流水号", observed=True).indices

        replayed = self.replay_log()
        # 全局重算 BMI（保证显示值与当前 weight/height 一致，而非 CSV 原始值）
        self._recalc_all_bmi()

        return {
            "rows": int(len(df)),
            "samples": int(len(self.sample_ids)),
            "replayed_edits": int(replayed),
        }

    def _recalc_all_bmi(self) -> None:
        """基于当前 weight 和 height 列实时重算全表 BMI，覆盖 CSV 中的原始值。"""
        if self.df is None or "weight" not in self.df.columns or "height" not in self.df.columns:
            return
        if "BMI" not in self.df.columns:
            self.df["BMI"] = np.nan
        for row_id in self.df.index:
            w = self.df.at[row_id, "weight"]
            h = self.df.at[row_id, "height"]
            if pd.notna(w) and pd.notna(h):
                h_val = float(h)
                h_m = h_val / 100.0 if h_val > 3 else h_val
                if h_m > 0:
                    self.df.at[row_id, "BMI"] = round(float(w) / (h_m * h_m), 2)
                    continue
            self.df.at[row_id, "BMI"] = np.nan

    def replay_log(self) -> int:
        if self.df is None or not self.log_path.exists():
            return 0

        log_df = pd.read_csv(self.log_path, low_memory=False)
        if log_df.empty:
            return 0

        required = {"row_id", "new_weight", "new_bmi", "new_deleted"}
        if not required.issubset(set(log_df.columns)):
            return 0

        # ---- 行数校验：检测 CSV 是否在两次加载之间被替换 ----
        has_row_count_col = "csv_row_count" in log_df.columns
        if has_row_count_col:
            log_row_counts = log_df["csv_row_count"].dropna().unique()
            if len(log_row_counts) > 0:
                log_rc = int(log_row_counts[0])
                if log_rc != self._csv_row_count:
                    print(
                        f"  [警告] edit log 记录时 CSV 行数={log_rc}，"
                        f"当前 CSV 行数={self._csv_row_count}。\n"
                        f"         CSV 可能已被重新生成！为安全起见，跳过全部 replay。\n"
                        f"         如确认需要应用旧日志，请手动删除: {self.log_path}"
                    )
                    return 0

        # ---- 逐条校验并 replay ----
        has_gday_col = "gestation_day" in log_df.columns
        applied = 0
        skipped = 0
        for row in log_df.itertuples(index=False):
            try:
                row_id = int(getattr(row, "row_id"))
            except Exception:
                continue
            if row_id not in self.df.index:
                skipped += 1
                continue

            # ---- 复合键校验：sample_id + gestation_day ----
            if has_gday_col:
                log_sid = str(getattr(row, "sample_id", "")).strip()
                log_gday = getattr(row, "gestation_day", None)
                actual_sid = str(self.df.at[row_id, "项目流水号"]).strip()
                actual_gday = self.df.at[row_id, "gestation_day"]
                sid_match = (log_sid == actual_sid)
                gday_match = True
                if log_gday is not None and not pd.isna(log_gday):
                    try:
                        gday_match = (int(float(log_gday)) == int(float(actual_gday)))
                    except (ValueError, TypeError):
                        gday_match = False
                if not sid_match or not gday_match:
                    print(
                        f"  [replay 跳过] row_id={row_id}: "
                        f"日志({log_sid}, day{log_gday}) ≠ "
                        f"实际({actual_sid}, day{actual_gday})"
                    )
                    skipped += 1
                    continue

            new_weight = getattr(row, "new_weight", np.nan)
            if not pd.isna(new_weight):
                self.df.at[row_id, "weight"] = float(new_weight)

            if "BMI" in self.df.columns:
                new_bmi = getattr(row, "new_bmi", np.nan)
                if not pd.isna(new_bmi):
                    self.df.at[row_id, "BMI"] = float(new_bmi)
                else:
                    self.df.at[row_id, "BMI"] = np.nan

            self.df.at[row_id, "_deleted"] = parse_bool(getattr(row, "new_deleted", False))
            applied += 1

        if skipped > 0:
            print(f"  [replay] 应用 {applied} 条，跳过 {skipped} 条（键不匹配或 row_id 越界）")
        return applied

    def get_sample_df(self, sample_id: str, include_deleted: bool = True) -> pd.DataFrame:
        if self.df is None:
            return pd.DataFrame()
        row_ids = self.sample_index.get(sample_id)
        if row_ids is None:
            return pd.DataFrame(columns=self.df.columns)
        sample_df = self.df.loc[row_ids].sort_values("gestation_day", kind="stable")
        if not include_deleted:
            sample_df = sample_df[~sample_df["_deleted"]]
        return sample_df

    def _recalc_bmi(self, row_id: int, new_weight: float) -> float:
        if self.df is None:
            return np.nan
        if "height" not in self.df.columns or "BMI" not in self.df.columns:
            return np.nan

        height_cm = self.df.at[row_id, "height"]
        if pd.isna(height_cm) or float(height_cm) <= 0:
            return np.nan
        height_m = float(height_cm) / 100.0
        return new_weight / (height_m * height_m)

    def _append_log(self, edit: EditResult) -> None:
        self.ensure_dirs()
        gday = np.nan
        if self.df is not None and edit.row_id in self.df.index:
            gday = self.df.at[edit.row_id, "gestation_day"]
        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operation": edit.operation,
            "row_id": edit.row_id,
            "sample_id": edit.sample_id,
            "gestation_day": gday,
            "old_weight": edit.old_weight,
            "new_weight": edit.new_weight,
            "old_bmi": edit.old_bmi,
            "new_bmi": edit.new_bmi,
            "old_deleted": edit.old_deleted,
            "new_deleted": edit.new_deleted,
            "csv_row_count": getattr(self, '_csv_row_count', np.nan),
        }
        frame = pd.DataFrame([payload], columns=LOG_COLUMNS)
        frame.to_csv(
            self.log_path,
            mode="a",
            index=False,
            header=not self.log_path.exists(),
            encoding="utf-8-sig",
        )

    def apply_operation(self, row_id: int, operation: str) -> EditResult:
        if self.df is None:
            raise RuntimeError("数据未加载。")
        if row_id not in self.df.index:
            raise KeyError(f"row_id 不存在: {row_id}")

        record = self.df.loc[row_id]
        sample_id = str(record["项目流水号"])
        old_weight = float(record["weight"])
        old_bmi = float(record["BMI"]) if "BMI" in self.df.columns and not pd.isna(record["BMI"]) else np.nan
        old_deleted = bool(record["_deleted"])

        if old_deleted and operation in {"x2", "div2", "add10", "sub10", "delete"}:
            raise ValueError("当前记录已删除。")

        if operation == "x2":
            new_weight = old_weight * 2.0
            new_deleted = False
        elif operation == "div2":
            new_weight = old_weight / 2.0
            new_deleted = False
        elif operation == "add10":
            new_weight = old_weight + 10.0
            new_deleted = False
        elif operation == "sub10":
            new_weight = old_weight - 10.0
            new_deleted = False
        elif operation == "delete":
            new_weight = old_weight
            new_deleted = True
        else:
            raise ValueError(f"未知操作: {operation}")

        if operation in {"x2", "div2", "add10", "sub10"} and (pd.isna(old_weight)):
            raise ValueError("当前体重为空，无法执行数值操作。")

        new_bmi = self._recalc_bmi(row_id, new_weight) if operation in {"x2", "div2", "add10", "sub10"} else old_bmi

        self.df.at[row_id, "weight"] = new_weight
        if "BMI" in self.df.columns:
            self.df.at[row_id, "BMI"] = new_bmi
        self.df.at[row_id, "_deleted"] = new_deleted

        result = EditResult(
            row_id=row_id,
            sample_id=sample_id,
            old_weight=old_weight,
            new_weight=new_weight,
            old_bmi=old_bmi,
            new_bmi=new_bmi,
            old_deleted=old_deleted,
            new_deleted=new_deleted,
            operation=operation,
        )
        self.history.append(result)
        self._append_log(result)
        return result

    def undo_last(self) -> Optional[EditResult]:
        if self.df is None or not self.history:
            return None

        last = self.history.pop()
        row_id = last.row_id
        if row_id not in self.df.index:
            return None

        current = self.df.loc[row_id]
        current_weight = float(current["weight"])
        current_bmi = (
            float(current["BMI"]) if "BMI" in self.df.columns and not pd.isna(current["BMI"]) else np.nan
        )
        current_deleted = bool(current["_deleted"])

        self.df.at[row_id, "weight"] = last.old_weight
        if "BMI" in self.df.columns:
            self.df.at[row_id, "BMI"] = last.old_bmi
        self.df.at[row_id, "_deleted"] = last.old_deleted

        undo = EditResult(
            row_id=row_id,
            sample_id=last.sample_id,
            old_weight=current_weight,
            new_weight=last.old_weight,
            old_bmi=current_bmi,
            new_bmi=last.old_bmi,
            old_deleted=current_deleted,
            new_deleted=last.old_deleted,
            operation=f"undo_{last.operation}",
        )
        self._append_log(undo)
        return undo

    def export_csv(self, output_path: Path) -> int:
        if self.df is None:
            raise RuntimeError("数据未加载。")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        export_df = self.df.loc[~self.df["_deleted"], self.original_columns]
        export_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        return int(len(export_df))

    def get_counts(self) -> Dict[str, int]:
        if self.df is None:
            return {"rows": 0, "deleted": 0, "active": 0}
        deleted = int(self.df["_deleted"].sum())
        total = int(len(self.df))
        return {"rows": total, "deleted": deleted, "active": total - deleted}


class ReviewApp:
    def __init__(self, root: tk.Tk, store: ReviewStore):
        self.root = root
        self.store = store

        self.current_sample_id: Optional[str] = None
        self.current_sample_df = pd.DataFrame()
        self.current_table_df = pd.DataFrame()
        self.current_active_df = pd.DataFrame()
        self.current_plot_df = pd.DataFrame()
        self.plot_row_ids: np.ndarray = np.array([], dtype=np.int64)
        self.selected_row_id: Optional[int] = None   # 最后一次单击/图表选中的主选点
        self.selected_row_ids: List[int] = []          # 当前全部选中点（支持多选）

        self.sample_pos_map: Dict[str, int] = {}

        self.sample_var = tk.StringVar()
        self.only_suspect_var = tk.BooleanVar(value=False)
        self.jump_threshold_var = tk.StringVar(value="8")
        self.sample_info_var = tk.StringVar(value="样本: -")
        self.selected_info_var = tk.StringVar(value="已选中记录: 无")
        self.status_var = tk.StringVar(value="准备就绪")

        self.tree: Optional[ttk.Treeview] = None
        self.row_item_map: Dict[int, str] = {}
        self.line = None

        self.fig = Figure(figsize=(9, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)

        self._build_ui()
        self._load_data()

    def _build_ui(self) -> None:
        self.root.title(APP_TITLE)
        self.root.geometry("1500x920")

        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(2, weight=1)

        ctrl = ttk.Frame(self.root, padding=(10, 8))
        ctrl.grid(row=0, column=0, columnspan=2, sticky="ew")
        ctrl.columnconfigure(1, weight=1)

        ttk.Label(ctrl, text="输入文件:").grid(row=0, column=0, sticky="w")
        self.source_label = ttk.Label(ctrl, text=str(self.store.source_path), foreground="#555555")
        self.source_label.grid(row=0, column=1, sticky="ew", padx=(6, 8))

        ttk.Button(ctrl, text="选择CSV", command=self._choose_source).grid(row=0, column=2, padx=4)
        ttk.Button(ctrl, text="重新加载", command=self._load_data).grid(row=0, column=3, padx=4)
        ttk.Button(ctrl, text="导出修正CSV", command=self._export_csv).grid(row=0, column=4, padx=4)

        sample_ctrl = ttk.Frame(self.root, padding=(10, 4))
        sample_ctrl.grid(row=1, column=0, columnspan=2, sticky="ew")
        sample_ctrl.columnconfigure(12, weight=1)

        ttk.Label(sample_ctrl, text="样本ID:").grid(row=0, column=0, sticky="w")
        sample_entry = ttk.Entry(sample_ctrl, textvariable=self.sample_var, width=22)
        sample_entry.grid(row=0, column=1, sticky="w", padx=(4, 6))
        sample_entry.bind("<Return>", lambda _: self._load_selected_sample())

        ttk.Button(sample_ctrl, text="加载样本", command=self._load_selected_sample).grid(row=0, column=2, padx=3)
        ttk.Button(sample_ctrl, text="上一样本", command=lambda: self._step_sample(-1)).grid(row=0, column=3, padx=3)
        ttk.Button(sample_ctrl, text="下一样本", command=lambda: self._step_sample(1)).grid(row=0, column=4, padx=3)
        ttk.Button(sample_ctrl, text="撤销上一步", command=self._undo_last).grid(row=0, column=5, padx=12)
        ttk.Checkbutton(
            sample_ctrl,
            text="仅显示疑似异常点",
            variable=self.only_suspect_var,
            command=self._refresh_current_sample_view,
        ).grid(row=0, column=6, padx=(4, 8), sticky="w")
        ttk.Label(sample_ctrl, text="跳变阈值(kg):").grid(row=0, column=7, sticky="e")
        threshold_entry = ttk.Entry(sample_ctrl, textvariable=self.jump_threshold_var, width=6)
        threshold_entry.grid(row=0, column=8, sticky="w", padx=(4, 2))
        threshold_entry.bind("<Return>", lambda _: self._refresh_current_sample_view())
        ttk.Button(sample_ctrl, text="应用阈值", command=self._refresh_current_sample_view).grid(
            row=0, column=9, padx=(2, 10)
        )

        ttk.Label(sample_ctrl, textvariable=self.sample_info_var, foreground="#1f3a56").grid(
            row=0, column=12, sticky="e"
        )

        plot_frame = ttk.Frame(self.root, padding=(10, 8))
        plot_frame.grid(row=2, column=0, sticky="nsew")
        plot_frame.rowconfigure(1, weight=1)
        plot_frame.columnconfigure(0, weight=1)

        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.grid(row=0, column=0, sticky="ew")

        canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas = canvas
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")

        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.LEFT)

        self.canvas.mpl_connect("pick_event", self._on_pick)

        table_frame = ttk.Frame(self.root, padding=(0, 8, 10, 8))
        table_frame.grid(row=2, column=1, sticky="nsew")
        table_frame.rowconfigure(1, weight=1)
        table_frame.columnconfigure(0, weight=1)

        ttk.Label(table_frame, textvariable=self.selected_info_var, foreground="#9c2c2c").grid(
            row=0, column=0, sticky="w", pady=(0, 6)
        )

        cols = ["row_id", "gestation_day", "weight", "BMI", "height", "type", "status", "suspect", "reason"]
        tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=24,
                            selectmode="extended")  # 支持 Shift/Ctrl 多选
        self.tree = tree
        self.row_item_map = {}

        headers = {
            "row_id": "row_id",
            "gestation_day": "gestation_day",
            "weight": "weight",
            "BMI": "BMI",
            "height": "height(cm)",
            "type": "type",
            "status": "status",
            "suspect": "suspect",
            "reason": "reason",
        }
        widths = {
            "row_id": 72,
            "gestation_day": 100,
            "weight": 88,
            "BMI": 80,
            "height": 80,
            "type": 110,
            "status": 72,
            "suspect": 72,
            "reason": 240,
        }

        for c in cols:
            tree.heading(c, text=headers[c])
            tree.column(c, width=widths[c], anchor="center", minwidth=50)

        yscroll = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        xscroll = ttk.Scrollbar(table_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        tree.grid(row=1, column=0, sticky="nsew")
        yscroll.grid(row=1, column=1, sticky="ns")
        xscroll.grid(row=2, column=0, sticky="ew")
        tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        tree.tag_configure("deleted", foreground="#888888")
        tree.tag_configure("suspicious", background="#fff4d6")

        # Shift+滚轮 → 横向滚动
        def _on_shift_scroll(event):
            # Windows: event.delta 为 ±120 的倍数
            # Linux/Mac: event.num 为 4/5
            if event.num == 4 or event.delta > 0:
                tree.xview_scroll(-2, "units")
            else:
                tree.xview_scroll(2, "units")
        tree.bind("<Shift-MouseWheel>", _on_shift_scroll)  # Windows / macOS
        tree.bind("<Shift-Button-4>", _on_shift_scroll)    # Linux 向上
        tree.bind("<Shift-Button-5>", _on_shift_scroll)    # Linux 向下

        action = ttk.LabelFrame(table_frame, text="点位操作", padding=(10, 8))
        action.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        action.columnconfigure(0, weight=1)
        action.columnconfigure(1, weight=1)
        action.columnconfigure(2, weight=1)
        action.columnconfigure(3, weight=1)
        action.columnconfigure(4, weight=1)

        self.btn_x2 = ttk.Button(action, text="×2", command=lambda: self._apply("x2"))
        self.btn_div2 = ttk.Button(action, text="÷2", command=lambda: self._apply("div2"))
        self.btn_add10 = ttk.Button(action, text="+10kg", command=lambda: self._apply("add10"))
        self.btn_sub10 = ttk.Button(action, text="-10kg", command=lambda: self._apply("sub10"))
        self.btn_delete = ttk.Button(action, text="删除该记录", command=lambda: self._apply("delete"))
        self.btn_x2.grid(row=0, column=0, sticky="ew", padx=4)
        self.btn_div2.grid(row=0, column=1, sticky="ew", padx=4)
        self.btn_add10.grid(row=0, column=2, sticky="ew", padx=4)
        self.btn_sub10.grid(row=0, column=3, sticky="ew", padx=4)
        self.btn_delete.grid(row=0, column=4, sticky="ew", padx=4)

        status = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w", padding=(8, 3))
        status.grid(row=3, column=0, columnspan=2, sticky="ew")

        self._set_action_state(False)
        self._init_plot()

    def _init_plot(self) -> None:
        self.ax.clear()
        self.ax.set_title("请选择样本")
        self.ax.set_xlabel("gestation_day")
        self.ax.set_ylabel("weight (kg)")
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw_idle()

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)
        self.root.update_idletasks()

    def _set_action_state(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        self.btn_x2.configure(state=state)
        self.btn_div2.configure(state=state)
        self.btn_add10.configure(state=state)
        self.btn_sub10.configure(state=state)
        self.btn_delete.configure(state=state)

    def _get_jump_threshold(self) -> float:
        raw = self.jump_threshold_var.get().strip()
        try:
            value = float(raw)
            if value <= 0:
                raise ValueError
            return value
        except Exception:
            self.jump_threshold_var.set("8")
            self._set_status("跳变阈值无效，已重置为 8 kg。")
            return 8.0

    def _compute_suspicious_for_active(self, active_df: pd.DataFrame) -> pd.DataFrame:
        if active_df.empty:
            return active_df

        threshold = self._get_jump_threshold()
        reviewed = active_df.sort_values("gestation_day", kind="stable").copy()
        weights = reviewed["weight"]
        prev_jump = (weights - weights.shift(1)).abs()
        next_jump = (weights.shift(-1) - weights).abs()
        out_of_range = (weights < 30) | (weights > 140)
        suspicious = out_of_range | (prev_jump >= threshold) | (next_jump >= threshold)

        reasons: List[str] = []
        for i in range(len(reviewed)):
            parts: List[str] = []
            if bool(out_of_range.iloc[i]):
                parts.append("体重超出[30,140]kg")
            p = prev_jump.iloc[i]
            n = next_jump.iloc[i]
            if pd.notna(p) and p >= threshold:
                parts.append(f"与前一条差{p:.1f}kg")
            if pd.notna(n) and n >= threshold:
                parts.append(f"与后一条差{n:.1f}kg")
            reasons.append("；".join(parts))

        reviewed["_suspicious"] = suspicious.fillna(False)
        reviewed["_suspect_reason"] = np.where(reviewed["_suspicious"], reasons, "")
        return reviewed

    def _refresh_current_sample_view(self) -> None:
        if self.current_sample_id:
            self._load_sample(self.current_sample_id, keep_selection=True)

    def _load_data(self) -> None:
        try:
            self._set_status("正在加载CSV，请稍候...")
            stats = self.store.load()
            self.source_label.configure(text=str(self.store.source_path))
            self.sample_pos_map = {sid: i for i, sid in enumerate(self.store.sample_ids)}

            if not self.store.sample_ids:
                self.sample_var.set("")
                self.sample_info_var.set("样本: 0/0")
                self._init_plot()
                self._set_status("加载完成，但没有可用样本ID。")
                return

            first = self.store.sample_ids[0]
            self.sample_var.set(first)
            self._load_sample(first)
            self._set_status(
                f"加载完成: {stats['rows']} 行, {stats['samples']} 个样本, 回放编辑 {stats['replayed_edits']} 条"
            )
        except Exception as exc:
            messagebox.showerror("加载失败", str(exc))
            self._set_status("加载失败")

    def _choose_source(self) -> None:
        path = filedialog.askopenfilename(
            title="选择输入CSV",
            filetypes=[("CSV 文件", "*.csv"), ("所有文件", "*.*")],
            initialdir=str(self.store.source_path.parent if self.store.source_path.exists() else self.store.work_dir),
        )
        if not path:
            return
        self.store.set_source_path(Path(path))
        self._load_data()

    def _load_selected_sample(self) -> None:
        sample = self.sample_var.get().strip()
        if not sample:
            return
        self._load_sample(sample)

    def _step_sample(self, step: int) -> None:
        if not self.store.sample_ids:
            return
        now = self.sample_var.get().strip()
        if now not in self.sample_pos_map:
            target = self.store.sample_ids[0]
        else:
            pos = self.sample_pos_map[now]
            target = self.store.sample_ids[max(0, min(len(self.store.sample_ids) - 1, pos + step))]
        self.sample_var.set(target)
        self._load_sample(target)

    def _load_sample(self, sample_id: str, keep_selection: bool = False) -> None:
        if sample_id not in self.sample_pos_map:
            messagebox.showwarning("样本不存在", f"未找到样本ID: {sample_id}")
            return

        previous_selected = self.selected_row_id if keep_selection else None
        self.current_sample_id = sample_id
        self.selected_row_id = None
        self.selected_row_ids = []
        self.selected_info_var.set("已选中记录: 无")
        self._set_action_state(False)

        sample_df = self.store.get_sample_df(sample_id, include_deleted=True).copy()
        sample_df["_suspicious"] = False
        sample_df["_suspect_reason"] = ""

        active_df = sample_df[~sample_df["_deleted"]].copy()
        active_marked = self._compute_suspicious_for_active(active_df)
        if not active_marked.empty:
            sample_df.loc[active_marked.index, "_suspicious"] = active_marked["_suspicious"]
            sample_df.loc[active_marked.index, "_suspect_reason"] = active_marked["_suspect_reason"]

        self.current_sample_df = sample_df
        self.current_active_df = sample_df[~sample_df["_deleted"]].copy()

        if self.only_suspect_var.get():
            focus_df = sample_df[(~sample_df["_deleted"]) & (sample_df["_suspicious"])].copy()
            self.current_table_df = focus_df
            self.current_plot_df = focus_df
        else:
            self.current_table_df = sample_df
            self.current_plot_df = self.current_active_df

        self._render_table()
        self._render_plot()

        total = len(sample_df)
        active = int((~sample_df["_deleted"]).sum()) if total else 0
        deleted = total - active
        suspicious = int(((~sample_df["_deleted"]) & (sample_df["_suspicious"])).sum()) if total else 0
        pos = self.sample_pos_map[sample_id] + 1
        mode_text = "疑似模式" if self.only_suspect_var.get() else "全量模式"
        self.sample_info_var.set(
            f"样本: {pos}/{len(self.store.sample_ids)}  记录: {active} (删除 {deleted})  疑似: {suspicious}  {mode_text}"
        )

        if previous_selected is not None:
            visible = (previous_selected in self.row_item_map) or (
                previous_selected in set(self.plot_row_ids.tolist())
            )
            if visible:
                self._select_row(previous_selected, sync_tree=True)

    def _render_table(self) -> None:
        if self.tree is None:
            return
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.row_item_map.clear()

        if self.current_table_df.empty:
            return

        for row_id, row in self.current_table_df.iterrows():
            row_id = int(row_id)
            deleted = bool(row["_deleted"])
            suspicious = bool(row.get("_suspicious", False))
            # height: 取该行 height 字段（cm 单位），若为 m 则自动转换显示
            raw_h = row.get("height", np.nan)
            if pd.notna(raw_h):
                h_val = float(raw_h)
                if 1.0 < h_val < 3.0:
                    h_val = round(h_val * 100, 1)  # m → cm
                height_str = f"{h_val:.1f}"
            else:
                height_str = ""
            # 实时 BMI：从当前 weight 和 height 现算，而非 CSV 原始值
            cur_w = self.store.df.at[row_id, "weight"] if self.store.df is not None and row_id in self.store.df.index else np.nan
            cur_h = self.store.df.at[row_id, "height"] if self.store.df is not None and row_id in self.store.df.index else np.nan
            if pd.notna(cur_w) and pd.notna(cur_h) and float(cur_h) > 0:
                h_m = float(cur_h) / 100.0 if float(cur_h) > 3 else float(cur_h)
                live_bmi = round(float(cur_w) / (h_m * h_m), 2) if h_m > 0 else np.nan
            else:
                live_bmi = np.nan
            values = [
                row_id,
                fmt_number(row.get("gestation_day", np.nan), 0),
                fmt_number(row.get("weight", np.nan), 3),
                fmt_number(live_bmi, 2),  # 实时 BMI
                height_str,
                "" if pd.isna(row.get("type", np.nan)) else str(row.get("type")),
                "deleted" if deleted else "active",
                "Y" if suspicious else "",
                str(row.get("_suspect_reason", "")) if suspicious else "",
            ]
            tags: List[str] = []
            if deleted:
                tags.append("deleted")
            if suspicious:
                tags.append("suspicious")
            item = self.tree.insert("", "end", values=values, tags=tuple(tags))
            self.row_item_map[row_id] = item

    def _render_plot(self) -> None:
        self.ax.clear()
        self.line = None
        only_suspect = self.only_suspect_var.get()
        active_df = self.current_active_df.sort_values("gestation_day", kind="stable")
        plot_df = self.current_plot_df.sort_values("gestation_day", kind="stable")
        self.plot_row_ids = (
            plot_df["_row_id"].to_numpy(dtype=np.int64) if not plot_df.empty else np.array([], dtype=np.int64)
        )

        sid = self.current_sample_id if self.current_sample_id is not None else "-"
        if active_df.empty:
            self.ax.text(0.5, 0.5, f"样本 {sid} 没有可显示记录", ha="center", va="center")
            self.ax.set_title(f"样本 {sid} 体重轨迹")
            self.ax.set_xlabel("gestation_day")
            self.ax.set_ylabel("weight (kg)")
            self.ax.grid(True, alpha=0.3)
            self.canvas.draw_idle()
            return

        ax_x = active_df["gestation_day"].to_numpy(dtype=float)
        ax_y = active_df["weight"].to_numpy(dtype=float)

        if only_suspect:
            # Keep full trajectory as context; only suspicious points are clickable/editable targets.
            self.ax.plot(
                ax_x,
                ax_y,
                color="#b8c0cc",
                marker="o",
                linestyle="-",
                markersize=4,
                linewidth=1.0,
                alpha=0.8,
            )
            if not plot_df.empty:
                x = plot_df["gestation_day"].to_numpy(dtype=float)
                y = plot_df["weight"].to_numpy(dtype=float)
                self.line = self.ax.plot(
                    x,
                    y,
                    color="#d68400",
                    marker="o",
                    linestyle="None",
                    markersize=8,
                    picker=6,
                )[0]
            else:
                self.ax.text(0.5, 0.95, "当前阈值下无疑似异常点", transform=self.ax.transAxes, ha="center", va="top")
        else:
            self.plot_row_ids = active_df["_row_id"].to_numpy(dtype=np.int64)
            self.line = self.ax.plot(
                ax_x,
                ax_y,
                color="#2f6f9f",
                marker="o",
                linestyle="-",
                markersize=6,
                linewidth=1.8,
                picker=6,
            )[0]
            sus_df = active_df[active_df["_suspicious"]]
            if not sus_df.empty:
                self.ax.scatter(
                    sus_df["gestation_day"].to_numpy(dtype=float),
                    sus_df["weight"].to_numpy(dtype=float),
                    s=75,
                    facecolor="#ffd36d",
                    edgecolor="#d68400",
                    linewidth=1.0,
                    zorder=4,
                )

        # 高亮所有选中点（多选时逐一标记）
        plot_ids_set = set(self.plot_row_ids.tolist())
        if self.selected_row_ids and self.line is not None:
            xdata = self.line.get_xdata()
            ydata = self.line.get_ydata()
            sel_x, sel_y = [], []
            for sel_rid in self.selected_row_ids:
                if sel_rid in plot_ids_set:
                    match = np.where(self.plot_row_ids == sel_rid)[0]
                    if len(match) > 0:
                        sel_x.append(xdata[int(match[0])])
                        sel_y.append(ydata[int(match[0])])
            if sel_x:
                self.ax.scatter(sel_x, sel_y, s=140, color="#d62728", zorder=6)

        mode_label = "疑似筛查模式" if only_suspect else "全量模式"
        self.ax.set_title(f"样本 {sid} 体重轨迹 ({mode_label})")
        self.ax.set_xlabel("gestation_day")
        self.ax.set_ylabel("weight (kg)")
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw_idle()

    def _on_pick(self, event) -> None:
        if self.line is None or event.artist != self.line:
            return
        if event.ind is None or len(event.ind) == 0:
            return
        idx = int(event.ind[0])
        if idx >= len(self.plot_row_ids):
            return
        row_id = int(self.plot_row_ids[idx])
        self._select_row(row_id, sync_tree=True)

    def _on_tree_select(self, _event) -> None:
        if self.tree is None:
            return
        picked = self.tree.selection()
        if not picked:
            return
        # 收集全部选中行
        row_ids = []
        for item in picked:
            vals = self.tree.item(item, "values")
            if vals:
                try:
                    row_ids.append(int(vals[0]))
                except (ValueError, TypeError):
                    pass
        if not row_ids:
            return
        # 以最后点击的行（picked[-1]）作为主选点
        self._select_rows(row_ids, primary_row_id=row_ids[-1], sync_tree=False)

    def _select_row(self, row_id: int, sync_tree: bool) -> None:
        """单点选中（保持向后兼容，内部委托给 _select_rows）。"""
        self._select_rows([row_id], primary_row_id=row_id, sync_tree=sync_tree)

    def _select_rows(self, row_ids: List[int], primary_row_id: int, sync_tree: bool) -> None:
        """多点选中核心方法。row_ids 为全部选中点，primary_row_id 为主选点（用于详情展示）。"""
        if self.store.df is None:
            return
        # 过滤掉不在 df 中的 id
        valid_ids = [r for r in row_ids if r in self.store.df.index]
        if not valid_ids:
            return
        self.selected_row_ids = valid_ids
        self.selected_row_id = primary_row_id if primary_row_id in self.store.df.index else valid_ids[0]

        if self.tree is not None and sync_tree:
            items = [self.row_item_map[r] for r in valid_ids if r in self.row_item_map]
            if items:
                self.tree.selection_set(items)
                self.tree.see(items[-1])

        row = self.store.df.loc[self.selected_row_id]
        deleted = bool(row["_deleted"])
        n = len(valid_ids)
        if n > 1:
            active_cnt = sum(
                1 for r in valid_ids
                if not bool(self.store.df.at[r, "_deleted"])
            )
            self.selected_info_var.set(
                f"已选中 {n} 条记录（{active_cnt} 条有效）| 按操作按钮批量执行"
            )
            # 只要有至少 1 条 active，就允许操作
            self._set_action_state(active_cnt > 0)
        else:
            status = "deleted" if deleted else "active"
            suspect_text = ""
            if self.selected_row_id in self.current_sample_df.index:
                sample_row = self.current_sample_df.loc[self.selected_row_id]
                if bool(sample_row.get("_suspicious", False)):
                    reason = str(sample_row.get("_suspect_reason", ""))
                    suspect_text = f" | suspect=Y {reason}" if reason else " | suspect=Y"
            self.selected_info_var.set(
                f"已选中 row_id={self.selected_row_id} | day={fmt_number(row['gestation_day'], 0)} | "
                f"weight={fmt_number(row['weight'], 3)} | status={status}{suspect_text}"
            )
            self._set_action_state(not deleted)
        self._render_plot()

    def _apply(self, operation: str) -> None:
        targets = [r for r in self.selected_row_ids if r in (self.store.df.index if self.store.df is not None else [])]
        if not targets:
            messagebox.showinfo("提示", "请先在轨迹图或表格中选中记录。")
            return

        results = []
        errors = []
        sample_id = None
        for row_id in targets:
            try:
                rec = self.store.df.loc[row_id]
                if bool(rec["_deleted"]) and operation in {"x2", "div2", "add10", "sub10", "delete"}:
                    continue  # 跳过已删除行
                result = self.store.apply_operation(row_id, operation)
                results.append(result)
                sample_id = result.sample_id
            except Exception as exc:
                errors.append(f"row_id={row_id}: {exc}")

        if errors:
            messagebox.showwarning("部分操作失败", "\n".join(errors[:5]))
        if not results:
            return

        if len(results) == 1:
            r = results[0]
            self._set_status(f"已执行 {operation}: row_id={r.row_id}, weight {r.old_weight:.3f} -> {r.new_weight:.3f}")
        else:
            self._set_status(f"已批量执行 {operation}: 共 {len(results)} 条")

        prev_ids = list(self.selected_row_ids)
        self._load_sample(sample_id, keep_selection=False)

        if operation == "delete":
            self.selected_row_ids = []
            self.selected_row_id = None
            self.selected_info_var.set("已选中记录: 无")
            self._set_action_state(False)
        else:
            # 重新选中仍可见的行
            still_visible = [r for r in prev_ids if r in self.row_item_map]
            if still_visible:
                self._select_rows(still_visible, primary_row_id=still_visible[-1], sync_tree=True)
            else:
                self.selected_row_ids = []
                self.selected_row_id = None
                self.selected_info_var.set("已选中记录: 无")
                self._set_action_state(False)
                if self.only_suspect_var.get():
                    self._set_status("选中点已修正，当前不再属于疑似列表。")

    def _undo_last(self) -> None:
        undo = self.store.undo_last()
        if undo is None:
            messagebox.showinfo("提示", "没有可撤销的操作。")
            return
        self._set_status(f"已撤销: row_id={undo.row_id}")
        self._load_sample(undo.sample_id, keep_selection=False)
        if (undo.row_id in self.row_item_map) or (undo.row_id in set(self.plot_row_ids.tolist())):
            self._select_row(undo.row_id, sync_tree=True)

    def _export_csv(self) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"{self.store.source_path.stem}_人工校正_{timestamp}.csv"
        path = filedialog.asksaveasfilename(
            title="导出修正结果",
            defaultextension=".csv",
            initialdir=str(self.store.export_dir),
            initialfile=default_name,
            filetypes=[("CSV 文件", "*.csv"), ("所有文件", "*.*")],
        )
        if not path:
            return

        try:
            rows = self.store.export_csv(Path(path))
            counts = self.store.get_counts()
            msg = (
                f"导出完成:\n{path}\n\n"
                f"保留记录: {rows}\n"
                f"累计标记删除: {counts['deleted']}\n"
                f"操作日志: {self.store.log_path}"
            )
            messagebox.showinfo("导出成功", msg)
            self._set_status(f"导出完成: {path}")
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc))


def build_default_source(script_dir: Path) -> Path:
    return (script_dir / DEFAULT_SOURCE_REL).resolve()


def run_headless_check(store: ReviewStore) -> None:
    stats = store.load()
    counts = store.get_counts()
    print("Loaded:", stats)
    print("Counts:", counts)
    if store.sample_ids:
        first = store.sample_ids[0]
        sample_df = store.get_sample_df(first)
        print(f"First sample: {first}, rows={len(sample_df)}")
    else:
        print("No sample IDs found.")


def main() -> None:
    parser = argparse.ArgumentParser(description="妊娠期体重人工排查工具")
    parser.add_argument(
        "--source",
        type=str,
        default="",
        help="输入CSV路径，默认使用 ../清洗管线重构_三步走/04_首尾异常与分娩校准版.csv",
    )
    parser.add_argument(
        "--headless-check",
        action="store_true",
        help="仅执行加载与基本检查，不打开图形界面",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    source_path = Path(args.source).resolve() if args.source else build_default_source(script_dir)
    store = ReviewStore(source_path=source_path, work_dir=script_dir)

    if args.headless_check:
        run_headless_check(store)
        return

    root = tk.Tk()
    ReviewApp(root, store)
    root.mainloop()


if __name__ == "__main__":
    main()
