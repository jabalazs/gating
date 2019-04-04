"""
Goal1: To be able to concurrently a Googl spreadsheet from different servers
       while keeping data consistency:
           Posible solutions:
               - Easy: Make each server write to a different sheet.
               - ???: get index of first empty row in sheet and use that while
                      experiment is running. The problem is that if two
                      processes access at the time they'll get the same id.

Goal2: To be able to receive a dict and a key and update corresponding entry
in Google Spreadsheet"""

from collections import OrderedDict

import gspread
from oauth2client.service_account import ServiceAccountCredentials

from .. import config

scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]


class GsheetsClient(object):
    def __init__(self):
        credentials = ServiceAccountCredentials.from_json_keyfile_name(
            config.JSON_KEYFILE_PATH, scope
        )
        gc = gspread.authorize(credentials)
        self.client = gc.open(config.SPREADSHEET_NAME)

    def worksheet(self, worksheet_name):
        gspread_wks = self.client.worksheet(worksheet_name)
        return Worksheet(gspread_wks)


class Worksheet(object):
    def auto_update(fn):
        def wrapper(self=None, *args, **kwargs):
            fn(self, *args, **kwargs)
            self.data_list = self.gsheet.get_all_values()

        return wrapper

    def __init__(self, worksheet: gspread.Worksheet):
        self.gsheet = worksheet
        self.data_list = worksheet.get_all_values()
        self.is_empty = True if len(self.data_list) == 0 else False

    def get_last_row_idx(self):
        return len(self.data_list)

    def get_last_col_idx(self):
        return len(self.data_list[-1])

    def get_first_row_idx(self):
        for idx, row in enumerate(self.data_list, start=1):
            if any(row):
                return idx

    def get_first_col_idx(self):
        # iterate over the elements of the last row
        for idx, elem in enumerate(self.data_list[-1], start=1):
            if elem:
                return idx

    def get_first_empty_cell_coords(self):
        return (self.get_last_row_idx() + 1, self.get_first_col_idx())

    def get_header(self):
        # Google sheet indices begin at 1
        header_row_idx = self.get_first_row_idx() - 1
        return self.data_list[header_row_idx][self.get_first_col_idx() - 1 :]

    @auto_update
    def insert_row_from_cell(self, row, row_idx, col_idx):
        grange = self.gsheet.range(
            row_idx, col_idx, row_idx, col_idx + len(row) - 1
        )
        assert len(grange) == len(row)
        for idx, elem in enumerate(row):
            grange[idx].value = elem
        self.gsheet.update_cells(grange)

    @auto_update
    def batch_insert_rows_from_cell(self, rows, row_idx, col_idx):
        num_rows = len(rows)
        len_rows = len(rows[0])
        assert all([len(row) == len_rows for row in rows]), "Malformed input"

        # Flat list of Cell instances
        grange = self.gsheet.range(
            row_idx, col_idx, row_idx + num_rows - 1, col_idx + len_rows - 1
        )
        assert len(grange) == num_rows * len_rows

        i = 0
        for row in rows:
            for elem in row:
                grange[i].value = elem
                i += 1

        self.gsheet.update_cells(grange)

    @auto_update
    def insert(self, datadict):
        ordered_dict = OrderedDict(sorted(datadict.items(), key=lambda t: t[0]))
        if self.is_empty:
            self.batch_insert_rows_from_cell(
                [ordered_dict.keys(), ordered_dict.values()], 1, 1
            )
            self.is_empty = False
        else:
            ins_row_idx, ins_col_idx = self.get_first_empty_cell_coords()
            header = self.get_header()
            if set(header) == set(ordered_dict.keys()):
                self.insert_row_from_cell(
                    ordered_dict.values(), ins_row_idx, ins_col_idx
                )

            else:
                print(
                    "Header is different from keys, beginning debugging session"
                )
                import ipdb

                ipdb.set_trace(context=10)


if __name__ == "__main__":

    gsheets = GsheetsClient()
    sheet = gsheets.worksheet("Sheet2")
    # import ipdb; ipdb.set_trace(context=10)
    # sheet.batch_insert_rows_from_cell([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 1, 1)
    # sheet.insert_row_from_cell(['elem1', 2, 'ASD'], 10, 5)
    sheet.insert({"a": 1, "b": 2, "c": 3, "y": "wasn't expecting this"})
    import ipdb

    ipdb.set_trace(context=10)
    sheet.insert({"a": "asd", "b": "qwe", "c": "hjk", "y": "lolololol"})
    sheet.insert({"a": 1, "b": 2, "r": 3, "y": "wasn't expecting this"})
