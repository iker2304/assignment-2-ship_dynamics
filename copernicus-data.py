import copernicusmarine

# Puntos P1 y P2 analizados
puntos = {
    "P1": {"lat": 53.055028, "lon": 4.127139},
    "P2": {"lat": 53.983056, "lon": 7.160472},
}

for nombre, coords in puntos.items():
    ds = copernicusmarine.open_dataset(
        dataset_id="cmems_mod_glo_wav_anfc_0.083deg_PT3H-i",
        variables=["VHM0", "VTPK", "VMDR"],  # Altura significativa, Periodo de pico y Dirección media
        minimum_longitude=coords["lon"],
        maximum_longitude=coords["lon"],
        minimum_latitude=coords["lat"],
        maximum_latitude=coords["lat"],
        start_datetime="2022-11-01T00:00:00",
        end_datetime="2026-11-26T00:00:00",
    )
    
    df = ds.to_dataframe().reset_index()
    archivo_salida = f"{nombre}_completo.csv"
    df.to_csv(archivo_salida, index=False)
    print(f"Guardado {archivo_salida} con {len(df)} registros.")