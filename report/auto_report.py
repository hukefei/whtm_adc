import numpy as np
import pandas as pd
import os

MAPPING = {
    'auto code': 'AOTU_CODE',
    'manu code': 'MANUAL_CODE',
    'confidence': 'CONFIDENCE',
    'auto flag': 'AUTO_FLAG',
    'product': 'PRODUCT_ID',
    'station': 'STEP_ID',
    'machine': 'EQP_ID',
    'tdi code': 'TDI_AUTO_CODE',
    'tdi confidence': 'TDI_CONFIDENCE'
}


class AutoReport(object):
    def __init__(self, data_path, mapping=MAPPING, sheet_name=0):
        self.data_path = data_path
        self.mapping = MAPPING
        self.with_product = True if (mapping.get('product', None) is not None) else False
        self.with_station = True if (mapping.get('station', None) is not None) else False
        self.with_machine = True if (mapping.get('machine', None) is not None) else False
        self.with_tdi = True if (mapping.get('tdi code', None) is not None) else False

        self.ori_data = self.read_data(data_path, sheet_name)
        self.cut_data = self.cut_data_df()

    @staticmethod
    def read_data(data_path, sheet_name=0):
        print('Reading data...')
        if data_path.endswith('xlsx'):
            data = pd.read_excel(data_path, sheet_name=sheet_name)
        elif data_path.endswith('csv'):
            data = pd.read_csv(data_path, sheet_name=sheet_name)
        else:
            raise Exception
        return data

    def cut_data_df(self):
        columns = []
        for k, v in self.mapping.items():
            if v is not None:
                columns.append(v)
        cut_data = self.ori_data.loc[:, columns]
        return cut_data

    def generate_report(self, save_file):
        print('Gernerating Report...')
        all_result = {}
        if self.with_station:
            station_table = self.cal_station()
            all_result.setdefault('站点统计表', station_table)
        if self.with_machine:
            machine_table = self.cal_machine()
            all_result.setdefault('机台统计表', machine_table)
        if self.with_product:
            product_table = self.cal_product()
            all_result.setdefault('产品统计表', product_table)
        code_table = self.cal_code()
        all_result.setdefault('code统计表', code_table)

        self.save_result(all_result, save_file)

    @staticmethod
    def save_result(results, save_file):
        writer = pd.ExcelWriter(save_file)
        for sheet, result in results.items():
            result.to_excel(writer, sheet)
        writer.save()
        writer.close()


    def cal_station(self):
        print('Calculating by station...')
        stations = self.cut_data[self.mapping['station']].unique()
        result = []
        for station in stations:
            df = self.cut_data[self.cut_data[self.mapping['station']] == station]
            r_ = self.basic_statics(df)
            r_['站点'] = station
            result.append(r_)
        result = pd.DataFrame(result)
        result = result.set_index('站点')
        return result

    def cal_machine(self):
        print('Calculating by machine...')
        machines = self.cut_data[self.mapping['machine']].unique()
        result = []
        for m in machines:
            df = self.cut_data[self.cut_data[self.mapping['machine']] == m]
            r_ = self.basic_statics(df)
            r_['机台'] = m
            result.append(r_)
        result = pd.DataFrame(result)
        result = result.set_index('机台')
        return result

    def cal_product(self):
        print('Calculating by product...')
        products = self.cut_data[self.mapping['product']].unique()
        result = []
        for p in products:
            df = self.cut_data[self.cut_data[self.mapping['product']] == p]
            r_ = self.basic_statics(df)
            r_['产品'] = p
            result.append(r_)
        result = pd.DataFrame(result)
        result = result.set_index('产品')
        return result

    def cal_code(self):
        print('Calculating by code...')
        codes = self.cut_data[self.mapping['auto code']].unique()
        result = []
        for c in codes:
            df = self.cut_data[self.cut_data[self.mapping['auto code']] == c]
            r_ = self.basic_statics(df)
            r_['code'] = c
            result.append(r_)
        result = pd.DataFrame(result)
        result = result.set_index('code')
        return result

    def basic_statics(self, df):
        data_all = len(df)
        data_out_model = sum(pd.isnull(df[self.mapping['auto code']]))
        data_model = data_all - data_out_model
        data_auto = len(df[df[self.mapping['auto flag']] == 1])
        data_manu = data_model - data_auto
        data_rejugde = len(df[(df[self.mapping['auto flag']] == 1) & ~pd.isnull(df[self.mapping['manu code']])])
        data_judge = len(df[(df[self.mapping['auto flag']] != 1) & ~pd.isnull(df[self.mapping['manu code']])])

        data_right_auto = len(df[(df[self.mapping['auto code']] == df[self.mapping['manu code']]) &
                                 (df[self.mapping['auto flag']] == 1)])
        data_right_filterd = len(df[(df[self.mapping['auto code']] == df[self.mapping['manu code']]) &
                                    (df[self.mapping['auto flag']] != 1)])

        data_thr_3 = len(df[df[self.mapping['confidence']] >= 0.3])
        data_thr_5 = len(df[df[self.mapping['confidence']] >= 0.5])
        data_thr_7 = len(df[df[self.mapping['confidence']] >= 0.7])
        data_thr_9 = len(df[df[self.mapping['confidence']] >= 0.9])

        cover_rate_system = data_model / data_all if data_all != 0 else 0
        cover_rate_model = data_auto / data_model if data_model != 0 else 0
        cover_rate_total = cover_rate_model * cover_rate_system
        acc_auto = data_right_auto / data_rejugde if data_rejugde != 0 else 1
        acc_filterd = data_right_filterd / data_judge if data_judge != 0 else 1

        if data_model != 0:
            ratio_3 = data_thr_3 / data_model
            ratio_5 = data_thr_5 / data_model
            ratio_7 = data_thr_7 / data_model
            ratio_9 = data_thr_9 / data_model
        else:
            ratio_3 = 0
            ratio_5 = 0
            ratio_7 = 0
            ratio_9 = 0

        return {'样本总数': data_all, '进入模型判图数': data_model, '系统转入人工数': data_out_model, '自动判图数': data_auto,
                '模型转人工数': data_manu, '系统覆盖率': '{:.1f}%'.format(cover_rate_system * 100),
                '模型覆盖率': '{:.1f}%'.format(cover_rate_model * 100), '总覆盖率': '{:.1f}%'.format(cover_rate_total * 100),
                '0.3阈值以上占比': '{:.1f}%'.format(ratio_3*100), '0.5阈值以上占比': '{:.1f}%'.format(ratio_5*100),
                '0.7阈值以上占比': '{:.1f}%'.format(ratio_7*100), '0.9阈值以上占比': '{:.1f}%'.format(ratio_9*100)}


if __name__ == '__main__':
    data_file = r'C:\Users\root\Desktop\121A4(1).xlsx'
    save_file = r'C:\Users\root\Desktop\auto_report_121A4(1).xlsx'
    ar = AutoReport(data_file)
    ar.generate_report(save_file)
