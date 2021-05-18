

# Basic Modules
import os
import sys
import numpy as np
# Read/Write to Excel
import pyexcel
import openpyxl as xl
# Openpyxl Styles
from openpyxl.styles import Alignment
from openpyxl.styles import Font



class excelProcessing:        
        
    def converToXLSX(self, testDataExcelFile):
        """
        Converts .xls Files to .xlsx Files That OpenPyxl Can Read
        If the File is Already a .xlsx Files, Do Nothing
        If the File is Neither a .xls Nor .xlsx, it Exits the Program
        """
        # Check That the Current Extension is .xls or .xlsx
        _, extension = os.path.splitext(testDataExcelFile)
        # If the Extension is .xlsx, the File is Ready; Do Nothing
        if extension == '.xlsx':
            return testDataExcelFile
        # If the Extension is Not .xls/.xlsx, Then the Data is in the Wrong Format; Exit Program
        if extension not in ['.xls', '.xlsx']:
            print("Cannot Convert File to .xlsx")
            sys.exit()
        
        # Create Output File Directory to Save Data ONLY If None Exists
        newExcelFolder = os.path.dirname(testDataExcelFile) + "/Excel Files/"
        os.makedirs(newExcelFolder, exist_ok = True)
        
        # Convert '.xls' to '.xlsx'
        filename = os.path.basename(testDataExcelFile)
        newExcelFile = newExcelFolder + filename + "x"
        pyexcel.save_as(file_name = testDataExcelFile, dest_file_name = newExcelFile, logfile=open(os.devnull, 'w'))
        
        # Save New Excel name
        return newExcelFile
    
    
    def getData(self, testDataExcelFile, testSheetNum = 0):
        """
        Extracts Pulse Data from Excel Document (.xlsx). Data can be in any
        worksheet which the user can specify using 'testSheetNum' (0-indexed).
        In the Worksheet:
            Time Data must be in Column 'A' (x-Axis)
            Capacitance Data must be in Column 'B' (y-Axis)
        If No Data is present in one cell of a row, it will be read in as zero.
        --------------------------------------------------------------------------
        Input Variable Definitions:
            testDataExcelFile: The Path to the Excel File Containing the Pulse Data
            testSheetNum: An Integer Representing the Excel Worksheet (0-indexed) Order.
        --------------------------------------------------------------------------
        """
        # Check if File Exists
        if not os.path.exists(testDataExcelFile):
            print("The following Input File Does Not Exist:", testDataExcelFile)
            sys.exit()
        # Convert to Exel if .xls Format; If .xlsx, Do Nothing; If Other, Exit Program
        testDataExcelFile = self.converToXLSX(testDataExcelFile)

        print("Extracting Data from the Excel File:", testDataExcelFile)
        # Load Data from the Excel File
        WB = xl.load_workbook(testDataExcelFile, data_only=True, read_only=True)
        WB_worksheets = WB.worksheets
        ExcelSheet = WB_worksheets[testSheetNum]
        
        # If Header Exists, Skip Until You Find the Data
        for row in ExcelSheet.rows:
            cellA = row[0]
            if type(cellA.value) == type(1.1):
                dataStartRow = cellA.row
                break
        
        data = dict(time=[], Capacitance=[])
        # Loop Through the Excel Worksheet to collect all the data
        for pointNum, [colA, colB] in enumerate(ExcelSheet.iter_rows(min_col=1, min_row=dataStartRow, max_col=2, max_row=ExcelSheet.max_row)):
            # Get Cell Values for First 4 Channels: Represents the Voltage for Each Channel;
            Time = colA.value; Capacitance = colB.value;
            
            # SafeGaurd: If User Edits the Document to Create Empty Rows, Stop Reading in Data
            if Time == None or Capacitance == None:
                break
            
            # Add Data to Dictionary
            data["time"].append(Time)
            data["Capacitance"].append(Capacitance)
             
        # Finished Data Collection: Close Workbook and Return Data to User
        print("Done Data Collecting"); WB.close()
        return np.array(data["time"]), np.array(data["Capacitance"])


    def saveResults(self, bloodPulse, pulseNumSaving, saveDataFolder, saveExcelName, sheetName = "Blood Pulse Data"):
        print("Saving the Data")
        # Create Output File Directory to Save Data: If Not Already Created
        os.makedirs(saveDataFolder, exist_ok=True)
        
        # Create Path to Save the Excel File
        excel_file = saveDataFolder + saveExcelName
        
        # If the File is Not Present: Create it
        if not os.path.isfile(excel_file):
            # Make Excel WorkBook
            WB = xl.Workbook()
            WB_worksheet = WB.active 
            WB_worksheet.title = sheetName
        else:
            print("Excel File Already Exists. Adding New Sheet to File")
            WB = xl.load_workbook(excel_file)
            WB_worksheet = WB.create_sheet(sheetName)
        
        # Label First Row
        header = ["Pulse Number", "Start Time", "End Time", 'Systolic Time From Start', 'Tidal Wave Time From Systolic', 'Dicrotic Time From Tidal Wave',
                  'Tail Wave Time From Dicrotic', 'End Time From Tail Wave', 'Systolic Peak Amplitude', 'Tidal Wave Peak Ampltiude',
                  "Dicrotic Peak Amplitude", 'Tail Wave Peak Ampltitude']
        header.extend(["", 'Gaussian Systolic Time From Start', 'Gaussian Tidal Wave Time From Systolic', 'Gaussian Dicrotic Time From Tidal Wave',
                  'Gaussian Tail Wave Time From Dicrotic', 'Gaussian End Time From Tail Wave', 'Gaussian Systolic Peak Amplitude', 'Gaussian Tidal Wave Peak Ampltiude',
                  "Gaussian Dicrotic Peak Amplitude", 'Gaussian Tail Wave Peak Ampltitude'])
        WB_worksheet.append(header)
        
        # Save Data to Worksheet
        for pulseNum in pulseNumSaving:
            # Write the Data to Excel
            WB_worksheet.append(bloodPulse[pulseNum]["Results to Save"])
        
        # Center the Data in the Cells
        align = Alignment(horizontal='center',vertical='center',wrap_text=True)        
        for column_cells in WB_worksheet.columns:
            length = max(len(str(cell.value) if cell.value else "") for cell in column_cells)
            WB_worksheet.column_dimensions[xl.utils.get_column_letter(column_cells[0].column)].width = length
            
            for cell in column_cells:
                cell.alignment = align
        # Header Style
        for cell in WB_worksheet["1:1"]:
            cell.font = Font(color='00FF0000', italic=True, bold=True)
            
        # Save as New Excel File
        WB.save(excel_file)
        WB.close()
      
        
        