// ROI Report: Report Information
// Processes the data returned from the application server 

import React, { useState } from "react"
import { Statistic } from 'semantic-ui-react'

export default function ROIReportInformation({ report, section_name, setAppStatus }) {

  // ** Validation **

  // Check report is available
  if (report == null ||
    report == undefined ||
    (!report.hasOwnProperty(section_name))) {
    return null;
  }

  // Get the report and properties
  const reportSection = report[section_name];
  const reportData = reportSection['data']
  const recordCount = reportData['record_count']
  const redundancy = reportData['redundancy']

  // Render the report
  return (
    <React.Fragment>
    <Statistic>
      <Statistic.Value>{recordCount}</Statistic.Value>
      <Statistic.Label>Total Records</Statistic.Label>
    </Statistic>
    <Statistic>
      <Statistic.Value>{redundancy}</Statistic.Value>
      <Statistic.Label>Total Sequences</Statistic.Label>
    </Statistic>
    </React.Fragment>
  );
}
