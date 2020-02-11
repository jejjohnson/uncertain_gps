import cdsapi

DATA_DIR = '/home/emmanuel/projects/2020_rbig_rs/data/climate/raw/'

def get_verify_data():

    c = cdsapi.Client()

    c.retrieve(
        'reanalysis-era5-single-levels-monthly-means',
        {
            'product_type':'monthly_averaged_reanalysis',
            'variable':[
                'mean_sea_level_pressure','surface_pressure'
            ],
            'year':[
                '1979','1980','1981',
                '1982','1983','1984',
                '1985','1986','1987',
                '1988','1989','1990',
                '1991','1992','1993',
                '1994','1995','1996',
                '1997','1998','1999',
                '2000','2001','2002',
                '2003','2004','2005',
                '2006','2007','2008',
                '2009','2010','2011',
                '2012','2013','2014',
                '2015','2016','2017',
                '2018','2019'
            ],
            'month':[
                '01','02','03',
                '04','05','06',
                '07','08','09',
                '10','11','12'
            ],
            'time':'00:00',
            'format':'netcdf'
        },
        f'{DATA_DIR}ERA5.nc')

    return None


def main():

    get_verify_data()

    return None
if __name__ == "__main__":
    main()
