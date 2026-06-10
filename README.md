# Sokoban
Single-file Python + pygame Sokoban. 20 BFS-verified levels, built-in solver, numpy-synthesized audio, procedural vector sprites, JSON persistence, state-machine architecture
# Sokoban

> Single-file Python + pygame Sokoban. 20 BFS-verified levels, built-in solver, numpy-synthesized audio, procedural vector sprites, JSON persistence, state-machine architecture.

A polished take on the classic warehouse-keeper puzzle: push crates onto their target tiles across 20 hand-crafted levels. Earn coins, unlock skins, climb per-level leaderboards, and use bombs or magic passes when you're stuck.

---

## Features

- **20 levels**, all verified solvable by the built-in BFS solver (optimal moves from 6 to 35).
- **Procedural graphics**: characters, crates and icons are drawn from primitives — no image assets required to run.
- **Synthesized audio**: all 10 sound effects (steps, push, win jingle, farm ambient loop, explosion…) are generated with numpy at startup.
- **Multi-player profiles** with persistent progress in a JSON save file.
- **Shop** with three tabs (characters, crates, consumable items) and a coin economy.
- **Two consumables**: bombs (destroy a wall) and magic passes (teleport a crate to a target).
- **Per-level leaderboards** ranked by moves, time and pushes; personal records per player.
- **Built-in solver** runs in a background thread and shows the level's optimum live in the HUD.
- **Audio settings** with separate music / SFX toggles and 11-step volume sliders.
- **Smooth movement animation** (110 ms ease-out) and undo stack (Z) up to 500 steps.
- **Optional assets** for richer visuals: `bg_menu.png` for the menu background and `pig_skin.png` for the pig sprite. If absent, the game falls back to procedural alternatives.

## Requirements

- Python 3.8+
- `pygame` (required)
- `numpy` (optional — if missing, the game runs without sound)

```bash
pip install pygame numpy
```

## Run

```bash
python sokoban_completo.py
```

The save file `sokoban_save.json` is created automatically next to the script.

## Controls

| Key | Action |
| --- | --- |
| `W A S D` / arrows | Move the player |
| `Z` | Undo last move |
| `R` | Restart current level |
| `B` | Activate bomb (then click a wall) |
| `M` | Activate magic pass (then click a crate, then a target) |
| `ESC` | Cancel item mode / exit to level select |
| Mouse | Navigate menus and buttons |

## Gameplay

Complete a level for the first time to earn coins (5 base, +20 every 10 levels, +40 for the optimal route, +100 for finishing the game). Replaying a level rewards +5 for beating your own move record and +5 for beating your own time. Coins are spent in the shop on new character skins, crate skins or consumables.

Using a bomb or a magic pass during a run disables the optimal-route and personal-record bonuses for that attempt, to keep leaderboards honest.

## Project structure

```
sokoban_completo.py        # Everything: engine, levels, solver, audio, UI
sokoban_arquitectura.pdf   # Architecture documentation (6 pages)
bg_menu.png                # Optional: menu background
pig_skin.png               # Optional: pig character sprite
sokoban_save.json          # Auto-generated save file
```

## Architecture overview

The game is built as a **finite state machine** with ten states (menu, name input, level select, play, win screens, ranking, shop, level-top leaderboard, settings). The main loop dispatches drawing and input to per-state handlers; buttons are registered each frame and resolved against mouse clicks by a generic dispatcher.

Top-level modules in the single source file:

| Block | Purpose |
| --- | --- |
| `SoundSystem` | Synthesizes the 10 game sounds with numpy oscillators + envelopes |
| `solve_level` | BFS solver returning the minimum-move solution |
| `make_*` functions | Procedural drawing of characters, crates, items and icons |
| `GameData` | JSON persistence layer (players, settings, leaderboards, optimum cache) |
| `get_levels` | The 20 levels as 9×9 character matrices |
| `SokobanGame` | Main class: state machine, screens, input, animation |


## Levels

All 20 levels are verified solvable by the BFS solver with a state cap of 1.5 million. Optimal moves grow roughly from 6 (level 1) to 35 (level 20). The cache is stored in the save file under `min_moves`, so the optimum is computed once per level and reused.

## Save file format

`sokoban_save.json` has four top-level keys:

- `players` — per-player profile (level, coins, equipped skins, unlocked items, records, inventory)
- `min_moves` — cached optima for each level
- `level_best` — per-level top-10 leaderboard
- `settings` — audio toggles and volumes

The loader merges saved settings with defaults so updating the game never breaks older save files.

## Notes

- The game starts in a 720×720 window. The grid is 9×9 with 80 px tiles.
- The solver runs on a background thread; it never blocks input.
- If `pygame.mixer` fails to initialize (e.g. headless environment), all audio calls become silent no-ops and the game keeps working.

## License

This project is provided as-is for personal and educational use. If you plan to distribute it publicly, make sure you have rights to any image assets you bundle alongside the script (`bg_menu.png`, `pig_skin.png`, etc.) — the procedural fallbacks in the source code are unencumbered.
