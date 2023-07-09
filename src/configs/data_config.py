DATA_CONFIG = {
    "datasets": [
        {
            "name": "IoTID20",
            "path": "data/IoTID20_final_max_1M.csv",
            "dropped_cols": [
                "Flow_ID",
                "Timestamp",
                "Src_IP",
                "Dst_IP",
                "Cat",
                "Sub_Cat",
            ],
            "target": "Target",
        },
        {
            "name": "CIC-DDoS-2019",
            "path": "data/CIC-DDoS-2019_final_max_1M.csv",
            "dropped_cols": [
                "Flow ID",
                "Source IP",
                "Destination IP",
                "Timestamp",
                "SimillarHTTP",
            ],
            "target": "Target",
        },
        {
            "name": "BOT-IoT",
            "path": "data/BOT-IoT_final_max_1M.csv",
            "dropped_cols": [
                "flgs",
                "proto",
                "saddr",
                "daddr",
                "state",
                "category",
                "subcategory",
            ],
            "target": "Target",
        },
        {
            "name": "CIC-IDS-2017",
            "path": "data/CIC-IDS-2017_final_max_1M.csv",
            "dropped_cols": ["Flow ID", "Source IP", "Destination IP", "Timestamp"],
            "target": "Target",
        },
        {
            "name": "CIC-IDS-2018",
            "path": "data/CIC-IDS-2018_final_max_1M.csv",
            "dropped_cols": ["Timestamp"],
            "target": "Target",
        },
    ],
}
