using System;
using System.Collections.Generic;

namespace Network
{
    // Server configuration, settings & options. Retrieve the latest config from NetworkManager.
    [Serializable]
    public class Config
    {
        // The time the config was fetched from the server. Generally a config
        // is considered stale if it is older than 1m.  NetworkManager is
        // responsible for keeping the config fresh.
        public DateTime timestamp = DateTime.MinValue;
        public string name = "";
        // public string data_prefix = "./";
        // public string record_diretory_suffix = "game_records/";
        // public string assets_directory_suffix = "assets/";
        // public string database_path_suffix = "game_data.db";
        // public string backup_db_path_suffix = "game_data.bk.db";
        // public List<List<int>> analysis_game_id_ranges;
        public int http_port = 8080;
        public bool gui = false;
        public int map_cache_size = 500;
        public string comment = "";
        // Card covers block the follower from seeing what pattern is on the
        // card. Leaders can still see card patterns.
        public bool card_covers = false;

        // Everything is 100% visible if it's closer than fog_start units away.
        public int fog_start = 13;
        // Everything is 100% opaque if it's farther than fog_end units away.
        public int fog_end = 20;
        // RE fog: everything in between fog_start and fog_end is linearly
        // interpolated to make a smooth transition.
    }
}