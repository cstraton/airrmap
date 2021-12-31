# Reporting functionality
# (generates the data for the reports in the user interface).

# tests -> test_reporting.py

# %%
import pandas as pd
from joblib import Parallel, delayed
from typing import Any, Dict, List, Optional

from airrmap.shared import models
from airrmap.shared.timing import Timing
import airrmap.application.seqlogo as seqlogo


def get_summary_report(df_records: pd.DataFrame,
                       facet_row_value: str,
                       facet_col_value: str,
                       seq_logo_cfgs: Optional[List[Dict[str, Any]]] = None,
                       cdr3_field: Optional[str] = None,
                       redundancy_field: Optional[str] = None,
                       v_field: Optional[str] = None,
                       d_field: Optional[str] = None,
                       j_field: Optional[str] = None,
                       seq_list_sample_size: int = 50,
                       t: Timing = Timing()) -> dict:
    """
    Get report data for the given records.

    parameters
    ----------
    df_records: pd.DataFrame
        The pre-filtered records for the selected region.

    facet_row_value: str
        Value of the selected facet, or '' if not selected.

    facet_col_value: str
        Value of the selected facet, or '' if not selected.

    seq_logo_cfgs: List[Dict[str, Any]] (optional)
        List of sequence logo configuration dictionaries.
        Each dictionary should contain (at a minimum):
            title: The title for the sequence logo in the report.
            gapped_field: The column containing the gapped sequence.
        Default is None.

    cdr3_field: str (optional)
        Name of the field in df_records containing the cdr3 amino acid sequence.

    redundancy_field: str (optional)
        Column containing the sequence redundancy (number of instances
        of a particular sequence).

    v_field: str (optional)
        Column containing the assigned V germline gene.

    d_field: str (optional)
        Column containing the assigned D germline gene.

    j_field: str (optional)
        Column containing the assigned J germline gene.

    seq_list_sample_size: int (optional)
        Size of the sample for the list of sequences,
        by default 50.

    t: Timing
        Timing instance for profiling, by default Timing().

    returns
    -------
    dict
        Dictionary containing:
        logo: The base 64 encoded plot (utf-8).
        seqs: The gapped sequence strings (dict).
        cdr3lengths: CDR3 lengths dictionary, where
            keys=field names and values=list of values for
            each field.
    """

    t.add('get_summary_report() started.')

    # Init
    result = dict(logo='',
                  seqs={0: 'No sequences found.'})
    seq_logo_cfgs = [] if seq_logo_cfgs is None else seq_logo_cfgs

    #
    # --- Enrichment and transformation ---
    #

    # Compute CDR3 length
    if cdr3_field is None:
        cdr3len_field = None
    else:
        cdr3len_field = '_cdr3_len'
        df_records[cdr3len_field] = df_records[cdr3_field].apply(
            len
        ).astype('uint16')
    t.add('Enrichment finished.', meta={"cdr3_field": cdr3_field})
    

    #
    # --- Produce the report ---
    #

    # If sequences found, then prepare report
    if len(df_records.index) > 0:
        # CDR3 lengths
        if (redundancy_field is not None) and (cdr3len_field is not None):
            cdr3_lengths: models.ReportItem = get_grouped_data(
                df_records=df_records,
                group_field=cdr3len_field,
                measure_field=redundancy_field,
                name='cdr3length',
                title='CDR3 Length',
                report_type='cdr3length',
                x_label='CDR3 Length',
                y_label='%',
                facet_row_value=facet_row_value,
                facet_col_value=facet_col_value,
                aggregate_method='sum'
            )
            result['cdr3_lengths'] = cdr3_lengths
            t.add('CDR3 report finished.')

        # V, D and J reports.
        field_list = (
            (v_field, 'vdist', 'V Distribution'),
            (d_field, 'ddist', 'D Distribution'),
            (j_field, 'jdist', 'J Distribution')
        )
        for field_item in field_list:
            if (field_item[0] is not None) and redundancy_field is not None:
                dargs = dict(
                    df_records=df_records,
                    group_field=field_item[0],
                    measure_field=redundancy_field,
                    name=field_item[1],
                    title=field_item[2],
                    report_type='germlinedist',
                    x_label='Gene',
                    y_label='%',
                    facet_row_value=facet_row_value,
                    facet_col_value=facet_col_value,
                    aggregate_method='sum',
                )

                subreport: models.ReportItem = get_grouped_data(**dargs)
                result[dargs['name']] = subreport
                t.add(dargs['title'] + ' report finished.')

        # Report information.
        report_info = get_report_info(
            df_records=df_records,
            redundancy_field=redundancy_field
        )
        result['report_info'] = report_info
        t.add('Report information finished.')

        # Flatten the numbered sequence field
        record_id_column = 'sys_point_id'

        # Get the sequence logo and example sequences
        logo_items = []

        # n_jobs=-1: all cpus, n_jobs=1, single cpu (useful for debugging)
        logo_items = Parallel(n_jobs=-1)(
            delayed(build_seq_logo)
            (
                logo_cfg['title'],
                logo_cfg['gapped_field'],
                df_records,
                redundancy_field
            )
            for logo_cfg in seq_logo_cfgs
        )
        result['logo'] = logo_items
        t.add('Sequence logo finished.')

    t.add('Produce report finished.')

    # Result
    return result


def build_seq_logo(title: str,
                   gapped_field: str,
                   df_records: pd.DataFrame,
                   redundancy_field: str):
    """
    Generate a sequence logo.

    Parameters
    ----------
    gapped_field : str
        Column containing the gapped sequence.

    title : str
        Title for the logo (e.g. CDR3).

    df_records : pd.DataFrame
        The pre-filtered records for the selected region.

    redundancy_field : str
        Column containing the sequence redundancy (number of instances
        of a particular sequence).

    Returns
    -------
    Dict
        Logo dictionary, containing the logo image
        and associated metadata.
    """

    # Init
    seqs_top = []
    seqs_bottom = []
    seqs_unique_count = 0

    # De-duplicate seqs and sum redundancy
    # TODO: Make top/bottom 5 configurable
    df_grouped_region = df_records.groupby(
        [gapped_field])[redundancy_field].sum().reset_index()
    seqs_top = df_grouped_region.nlargest(
        5, redundancy_field).values.tolist()
    seqs_bottom = df_grouped_region.nsmallest(
        5, redundancy_field).values.tolist()

    # Squeeze gapped sequence (remove excess gaps)
    seqs_top = squeeze_gapped_seqs(seqs_top, min_gap_chars=0)
    seqs_bottom = squeeze_gapped_seqs(seqs_bottom, min_gap_chars=0)

    # Add rest of gapped seqs data to report
    seqs_unique_count = len(df_grouped_region.index)
    logo_item = dict(
        seqs_top=seqs_top,  # type: ignore
        seqs_bottom=seqs_bottom,  # type: ignore
        seqs_unique_count=seqs_unique_count,  # type: ignore
        region_name=title
    )

    # Create logo
    logo, df_logo_counts = seqlogo.get_logo(
        gapped_seqs=df_records[gapped_field].values,
        weights=df_records[redundancy_field].values,
        title='',  # render client side using logo_item.region_name
        encode_base64=True,
        image_format='png')
    logo_item['logo_img'] = logo

    # Return
    return logo_item


def get_report_info(df_records: pd.DataFrame,
                    redundancy_field: Optional[str] = None) -> models.ReportItem:
    """
    Get general information about the report

    Parameters
    ----------
    df_records : pd.DataFrame
        The pre-filtered records for the report.

    redundancy_field : str (optional)
        The field containing the sequence redundancy number.


    Returns
    -------
    models.ReportItem
        [description]
    """

    record_count = len(df_records.index)
    redundancy = int(df_records[redundancy_field].sum()
                     ) if redundancy_field is not None else 0
    report_data = dict(
        record_count=record_count,
        redundancy=redundancy
    )

    report = models.ReportItem(
        name='reportinfo',
        title='Report Information',
        report_type='reportinfo',
        data=report_data
    )

    return report


def get_grouped_data(df_records: pd.DataFrame,
                     group_field: str,
                     measure_field: str,
                     name: str,
                     title: str,
                     report_type: str,
                     x_label: str,
                     y_label: str,
                     facet_row_value: str,
                     facet_col_value: str,
                     aggregate_method: str = 'sum') -> models.ReportItem:
    """
    Groups selected data by facet row/col and given field.

    Produces the data for a plot in the front-end.

    Parameters
    ----------
    df_records : pd.DataFrame
        The pre-filtered records.

    group_field: str
        The field to group by in addition to facet_row and facet_col.

    measure_field: str
        The field containing the value we're measuring.

    name: str
        Unique name for this report instance.

    title: str
        Title for this report (may be used in plots).

    report_type: str
        Report type.

    x_label: str
        Label for the x-axis (for plotting).

    y_label: str
        Label for the y-axis (for plotting).

    facet_row_value: str
        Value of the selected facet, or '' if not selected.

    facet_col_value: str
        Value of the selected facet, or '' if not selected.

    aggregate_method: str
        Pandas aggregation method to apply, by default 'sum'.

    Returns
    -------
    model.ReportItem
        Example:
        {
            name: 'cdr3length_1'
            title: 'CDR3 Length'
            report_type: 'cdr3length',
            x_label: 'CDR3 Length',
            y_label: '%',
            data:
                [
                    {
                        facet_row_value: 'Fv_volunteer',
                        facet_col_value: 'after-Day-1',
                        group_values: [2, 3, 4],
                        measure_values: [10, 20, 20],
                        measure_pcts: [20.0, 40.0, 40.0],
                    }
                ]
        }
    """

    NOT_SPECIFIED = ''

    # Aggregate measure by facets and group field
    df_grouped = df_records.groupby(
        ['facet_row', 'facet_col', group_field]
    )

    df_grouped = df_grouped.agg(
        {measure_field: aggregate_method}
    )

    # TODO:
    # Bug workaround (Pandas issue?): For some reason, applying the .agg()
    # method above produces a GroupBy object with -all-
    # facet rows and columns instead of just the ones in the original
    # GroupBy object. As a workaround, we re-filter...
    df_grouped.reset_index(inplace=True)
    if facet_row_value != NOT_SPECIFIED:
        df_grouped = df_grouped[df_grouped['facet_row'] == facet_row_value]
    if facet_col_value != NOT_SPECIFIED:
        df_grouped = df_grouped[df_grouped['facet_col'] == facet_col_value]

    # Compute pct, grouped by facet. Each facet should sum to 100.0.
    measure_field_pct = measure_field + '_pct'
    df_grouped[measure_field_pct] = 100 * (df_grouped[measure_field] / df_grouped.groupby(
        ['facet_row', 'facet_col']
    )[measure_field].transform(aggregate_method))

    # If no sequences for a given facet region, will be nan due to
    # divide by zero. Fill nans with zero, otherwise sum() will result in nan and
    # cause an issue with JSON encoding/decoding (nan not a valid value).
    df_grouped[measure_field_pct] = df_grouped[measure_field_pct].fillna(0)

    # Reset index
    df_grouped.reset_index(inplace=True)

    # Get list of facets (row and col)
    # Designed for loading into Plotly.js chart
    # See: https://plotly.com/javascript/react/

    # Leave facet sort order to front end
    # to be consistent with rendered tiles.

    # See docstring for example result.

    # NOTE!: If changing field names, front-end code will
    #        need to be updated. Facet values may not always be text.
    #       (if considering changing to dictionary)
    df_facets = df_grouped.groupby(['facet_row', 'facet_col'])
    facet_records = []
    for x in df_facets.groups:
        df_facet_single = df_facets.get_group(x)
        facet_row_value, facet_col_value = x  # Group will contain tuple
        data_values = df_facet_single[[
            group_field, measure_field, measure_field_pct]].to_dict(orient='list')

        # Convert numpy data types to native Python (workaround)
        # otherwise JSON.dumps will fail later as doesn't
        # support numpy types.
        # Possibly related to issue: https://github.com/pandas-dev/pandas/issues/16048
        # If numpy datatype, use .item() to convert to its native Python type
        for i in range(len(data_values[measure_field])):
            measure_value = data_values[measure_field][i]
            if type(measure_value).__module__ == 'numpy':
                data_values[measure_field][i] = measure_value.item()

        # Add the facet data
        facet_records.append(
            dict(
                facet_row_value=facet_row_value,
                facet_col_value=facet_col_value,
                group_values=data_values[group_field],
                measure_values=data_values[measure_field],
                measure_pcts=data_values[measure_field_pct]

            )
        )

    # Build the final report
    report = models.ReportItem(
        name=name,
        title=title,
        report_type=report_type,
        x_label=x_label,
        y_label=y_label,
        data=facet_records
    )

    # Return the result
    return report


def convert_list_numpy(lst):

    # If numpy datatype, use .item() to convert to its native Python type.
    # Used for JSON-encoding as JSON doesn't support numpy data types.
    lst = [x.item() for x in lst if x.__module__ == 'numpy']


def squeeze_gapped_seqs(gapped_seq_list: List[List], min_gap_chars=2, gap_char='.') -> List:
    """
    Squeezes gapped strings / removes excess gaps.

    Assumes all gaps for each sequence are always together
    (i.e. the center of the sequence).

    Parameters
    ----------
    gapped_seq_list : List[List]
        List of gapped strings. Items should be
        a list of 2 elements:
        [gapped_sequence, measure_value].

    min_gap_chars : int, optional
        Minimum number of gap characters that should remain after
        being squeezed. If any sequences had less gaps
        originally, this has no effect. By default, 2.

    gap_char : str, optional
        The gap character, by default '.'

    Returns
    -------
    List
        Either the original list if no changes were made,
        or the sequences with excess gaps removed.
        Note the latter will return a list of tuples due
        to use of the zip function.
    """

    SEQ_ELEMENT = 0
    MEASURE_ELEMENT = 1

    # First pass, get the minimum number of consecutive gap characters
    # across all sequences.
    min_consecutive_gaps_found = 888
    for gapped_seq in gapped_seq_list:
        max_consecutive_gaps_found = 0
        consecutive_gaps = 0
        for char in gapped_seq[SEQ_ELEMENT]:  # ['GGG..GG', 1] seq,redundancy
            if char == gap_char:
                consecutive_gaps += 1
                if consecutive_gaps > max_consecutive_gaps_found:
                    max_consecutive_gaps_found = consecutive_gaps
            else:
                consecutive_gaps = 0
        if max_consecutive_gaps_found < min_consecutive_gaps_found:
            min_consecutive_gaps_found = max_consecutive_gaps_found

    # Remove excess gaps (find-replace only once, not all occurrences)
    num_gaps_to_remove = min_consecutive_gaps_found - min_gap_chars
    if num_gaps_to_remove > 0:
        find_text = gap_char * num_gaps_to_remove
        seqs_adjusted = [seq[SEQ_ELEMENT].replace(
            find_text, '', 1) for seq in gapped_seq_list]
        measure_values = [x[MEASURE_ELEMENT] for x in gapped_seq_list]
        assert len(seqs_adjusted) == len(
            measure_values), 'Number of sequences and measure values should be the same.'
        return list(zip(seqs_adjusted, measure_values))

    else:
        return gapped_seq_list
