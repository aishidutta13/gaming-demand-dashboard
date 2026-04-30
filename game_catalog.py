GAME_APPIDS = {
    "dota 2": 570, "dota": 570, "dota2": 570,
    "csgo": 730, "cs2": 730, "cs": 730,
    "counter strike": 730,
    "counter strike 2": 730,
    "counter strike global offensive": 730,
    "counter strike: global offensive": 730,
    "pubg": 578080,
    "pubg battlegrounds": 578080,
    "pubg: battlegrounds": 578080,
    "apex": 1172470,
    "apex legends": 1172470,
    "gta": 271590,
    "gta v": 271590,
    "gta 5": 271590,
    "grand theft auto v": 271590,
    "destiny 2": 1085660,
    "destiny2": 1085660,

    "7 days to die": 251570,
    "age of empires ii: definitive edition": 813780,
    "ark: survival evolved": 346110,
    "arma 3": 107410,
    "battlefield 1": 1238840,
    "dead by daylight": 381210,
    "dying light": 239140,
    "dyson sphere program": 1366540,
    "ea sports fifa 21": 1313860,
    "efootball pes 2021 season update": 1259970,
    "eternal return": 1049590,
    "football manager 2021": 1263850,
    "rust": 252490,
    "team fortress 2": 440,
    "warframe": 230410,
    "tom clancy's rainbow six siege": 359550,
    "rainbow six siege": 359550,
    "terraria": 105600,
    "stardew valley": 413150,
    "war thunder": 236390,
    "monster hunter: world": 582010,
    "path of exile": 238960,
    "the sims 4": 1222670,
    "lost ark": 1599340,
    "wallpaper engine": 431960
}

ALIASES = {
    "dota": "dota 2",
    "dota2": "dota 2",
    "cs": "counter strike: global offensive",
    "csgo": "counter strike: global offensive",
    "cs2": "counter strike: global offensive",
    "counter strike": "counter strike: global offensive",
    "counter strike global offensive": "counter strike: global offensive",
    "pubg": "pubg: battlegrounds",
    "pubg battlegrounds": "pubg: battlegrounds",
    "apex": "apex legends",
    "gta": "grand theft auto v",
    "gta v": "grand theft auto v",
    "gta 5": "grand theft auto v",
    "destiny2": "destiny 2",
    "rainbow six siege": "tom clancy's rainbow six siege"
}

SPECIAL_TITLES = {
    "dota 2": "Dota 2",
    "counter strike: global offensive": "Counter Strike: Global Offensive",
    "pubg: battlegrounds": "PUBG: Battlegrounds",
    "apex legends": "Apex Legends",
    "grand theft auto v": "Grand Theft Auto V",
    "destiny 2": "Destiny 2",
    "7 days to die": "7 Days to Die",
    "age of empires ii: definitive edition": "Age of Empires II: Definitive Edition",
    "ark: survival evolved": "ARK: Survival Evolved",
    "arma 3": "Arma 3",
    "battlefield 1": "Battlefield 1",
    "dead by daylight": "Dead by Daylight",
    "dying light": "Dying Light",
    "dyson sphere program": "Dyson Sphere Program",
    "ea sports fifa 21": "EA Sports FIFA 21",
    "efootball pes 2021 season update": "eFootball PES 2021 Season Update",
    "eternal return": "Eternal Return",
    "football manager 2021": "Football Manager 2021",
    "rust": "Rust",
    "team fortress 2": "Team Fortress 2",
    "warframe": "Warframe",
    "tom clancy's rainbow six siege": "Tom Clancy's Rainbow Six Siege"
}

GAME_CATEGORIES = {
    "dota 2": "MOBA",
    "counter strike: global offensive": "FPS",
    "pubg: battlegrounds": "Battle Royale",
    "apex legends": "Battle Royale",
    "grand theft auto v": "Open World",
    "destiny 2": "FPS",
    "7 days to die": "Survival",
    "age of empires ii: definitive edition": "Strategy",
    "ark: survival evolved": "Survival",
    "arma 3": "Tactical",
    "battlefield 1": "FPS",
    "dead by daylight": "Survival",
    "dying light": "Survival",
    "dyson sphere program": "Simulation",
    "ea sports fifa 21": "Sports",
    "efootball pes 2021 season update": "Sports",
    "eternal return": "MOBA",
    "football manager 2021": "Simulation",
    "rust": "Survival",
    "team fortress 2": "FPS",
    "warframe": "Action RPG",
    "tom clancy's rainbow six siege": "FPS",
    "terraria": "Sandbox",
    "stardew valley": "Simulation",
    "war thunder": "Simulation",
    "monster hunter: world": "Action RPG",
    "path of exile": "Action RPG",
    "the sims 4": "Simulation",
    "lost ark": "MMO",
    "wallpaper engine": "Utility"
}


def normalize_game_name(game):
    game = str(game).lower().strip()
    return ALIASES.get(game, game)


def title_game(game):
    normalized = normalize_game_name(game)

    if normalized in SPECIAL_TITLES:
        return SPECIAL_TITLES[normalized]

    return " ".join(word.capitalize() for word in normalized.split())


def canonical_game_appids():
    appids = {}

    for game, appid in GAME_APPIDS.items():
        appids[normalize_game_name(game)] = appid

    return dict(sorted(appids.items()))


def steam_header_url(appid):
    return f"https://cdn.cloudflare.steamstatic.com/steam/apps/{appid}/header.jpg"


def game_category(game):
    normalized = normalize_game_name(game)
    return GAME_CATEGORIES.get(normalized, "General")
