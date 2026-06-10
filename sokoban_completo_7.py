"""
================================================================================
 SOKOBAN — Juego completo en Python + pygame
================================================================================

Sokoban ("almacenero" en japonés) es un puzzle clásico: el jugador empuja cajas
por un almacén hasta colocarlas todas sobre sus casillas-objetivo. Las cajas solo
se pueden EMPUJAR (nunca tirar) y solo de una en una.

Este archivo es autocontenido: incluye el motor del juego, los 20 niveles, un
resolvedor automático (solver), generación procedural de gráficos y sonidos,
persistencia en disco, tienda, ranking, sistema de items y todas las pantallas.

ESTRUCTURA GENERAL DEL ARCHIVO (en orden de aparición):
  1. Imports y configuración global ............ constantes, paleta de color
  2. SoundSystem .............................. sonidos sintetizados con numpy
  3. solve_level .............................. resolvedor BFS de Sokoban
  4. Generadores de skins ..................... dibujan personajes y cajas
  5. Generadores de items ..................... iconos de bomba y varita
  6. GameData ................................. persistencia en JSON
  7. get_levels ............................... definición de los 20 niveles
  8. SokobanGame .............................. clase principal: bucle y estados

DEPENDENCIAS:
  - pygame : ventana, gráficos, eventos, audio (obligatoria)
  - numpy  : síntesis de sonido (opcional; si falta, el juego corre sin audio)

ARCHIVOS EXTERNOS OPCIONALES (mismo directorio que este script):
  - bg_menu.png  : imagen de fondo del menú principal
  - pig_skin.png : sprite del personaje "cerdo"
  - sokoban_save.json : se crea solo; guarda jugadores, progreso y ajustes
================================================================================
"""

# ── Imports de la biblioteca estándar ─────────────────────────────────────────
import pygame as pg          # motor multimedia: ventana, dibujo, eventos, audio
import sys                   # para sys.exit() al cerrar el juego
import json                  # serialización del archivo de guardado
import os                    # comprobar existencia de archivos, rutas
import math                  # trigonometría para dibujar iconos y partículas
import random                # posiciones de estrellas, partículas de explosión
import threading             # el solver corre en un hilo aparte (no bloquea)
from collections import deque  # cola FIFO eficiente para la búsqueda BFS

# numpy es opcional: solo se usa para sintetizar sonidos. Si no está instalado,
# HAVE_NUMPY queda en False y el SoundSystem se desactiva silenciosamente.
try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

# ── Configuración global ──────────────────────────────────────────────────────
# Estas constantes definen el tamaño del juego y no cambian en tiempo de ejecución.
SAVE_FILE = 'sokoban_save.json'   # nombre del archivo de guardado en disco
WINDOW_SIZE = 720                 # lado de la ventana en píxeles (cuadrada)
GRID = 9                          # el tablero es de 9x9 celdas
TILE = WINDOW_SIZE // GRID        # tamaño de cada celda = 80 px
SAMPLE_RATE = 44100               # frecuencia de muestreo del audio (Hz, calidad CD)
MOVE_ANIM_MS = 110                # duración de la animación de un paso (ms)

# Paleta de color base (tuplas RGB). Centralizada aquí para mantener coherencia
# visual en todas las pantallas.
BG_DARK   = (18, 18, 28)     # fondo general oscuro
BG_PANEL  = (28, 28, 42)     # fondo de paneles y botones
PRIMARY   = (90, 160, 255)   # azul de acento (botón principal)
ACCENT    = (255, 200, 80)   # amarillo dorado (monedas, resaltados)
GOOD      = (90, 220, 130)   # verde (éxito, acción positiva)
BAD       = (230, 90, 90)    # rojo (error, peligro, salir)
TEXT      = (235, 235, 240)  # texto principal (casi blanco)
TEXT_DIM  = (160, 160, 175)  # texto secundario (gris claro)
LOCKED    = (70, 70, 85)     # gris de elementos bloqueados/inactivos


# ── Sistema de sonidos sintetizados ───────────────────────────────────────────
class SoundSystem:
    """Sintetiza todos los efectos de sonido del juego en memoria usando numpy
    (sin archivos de audio). Cada sonido es una onda calculada matemáticamente.

    Diseño defensivo: si numpy no está, o el mixer de pygame no se puede
    inicializar (p. ej. en un servidor sin tarjeta de sonido), `self.ok` queda
    en False y TODOS los métodos públicos se vuelven no-op. El juego sigue
    funcionando con normalidad, simplemente en silencio."""

    def __init__(self):
        self.ok = False             # True solo si el audio se inicializa bien
        self.sounds = {}            # dict {nombre: pygame.Sound}
        self.loop_channel = None    # canal reservado para la música en bucle
        # Volúmenes y conmutadores; modificables en tiempo real desde Ajustes.
        self.music_volume = 0.5
        self.sfx_volume = 0.6
        self.music_enabled = True
        self.sfx_enabled = True
        # Nombres que se tratan como "música" (se ven afectados por music_volume
        # en vez de sfx_volume). El resto son efectos puntuales.
        self._music_names = {'farm_ambient'}
        # Sin numpy no se puede sintetizar nada: abortamos dejando ok=False.
        if not HAVE_NUMPY:
            return
        try:
            # pre_init configura el mixer ANTES de abrirlo: 44.1kHz, 16-bit con
            # signo, estéreo (2 canales), buffer de 512 muestras (baja latencia).
            pg.mixer.pre_init(SAMPLE_RATE, -16, 2, 512)
            pg.mixer.init()
        except pg.error:
            return  # no hay dispositivo de audio: seguimos en silencio
        if pg.mixer.get_init() is None:
            return
        self.ok = True
        self._build_all()   # genera y cachea todos los sonidos de una vez

    # ---------- generadores básicos de forma de onda ----------
    # Estos métodos devuelven arrays numpy de float32 en el rango [-1, 1] que
    # representan audio mono. Se combinan luego para formar sonidos complejos.

    def _silence(self, ms):
        """Devuelve `ms` milisegundos de silencio (array de ceros)."""
        n = int(SAMPLE_RATE * ms / 1000)
        return np.zeros(n, dtype=np.float32)

    def _tone(self, freq, ms, *, decay=8, attack_ms=2, wave='sine', vib=0.0):
        """Genera un tono de frecuencia fija con envolvente de caída exponencial.

        freq      : frecuencia en Hz (la altura de la nota)
        ms        : duración en milisegundos
        decay     : velocidad de extinción del sonido (mayor = se apaga antes)
        attack_ms : rampa de subida inicial para evitar 'clicks' al empezar
        wave      : forma de onda ('sine', 'square', 'saw' o 'tri')
        vib       : profundidad de vibrato (0 = sin vibrato)
        """
        n = int(SAMPLE_RATE * ms / 1000)                       # nº de muestras
        t = np.linspace(0, ms/1000, n, False, dtype=np.float32)  # eje temporal
        # La fase del oscilador; con vibrato se modula a 6 Hz.
        if vib:
            phase = 2*np.pi*freq*t + vib*np.sin(2*np.pi*6*t)
        else:
            phase = 2*np.pi*freq*t
        # Forma de onda elegida:
        if wave == 'sine':
            s = np.sin(phase)                       # suave, pura
        elif wave == 'square':
            s = np.sign(np.sin(phase))              # dura, tipo chiptune
        elif wave == 'saw':
            s = 2*((freq*t) % 1) - 1                # brillante, áspera
        elif wave == 'tri':
            s = 2*np.abs(2*((freq*t)%1) - 1) - 1    # intermedia
        else:
            s = np.sin(phase)
        # Envolvente: caída exponencial (el sonido se desvanece).
        env = np.exp(-decay*t)
        # Rampa de ataque: los primeros `attack_ms` suben de 0 a 1 linealmente
        # para que el sonido no empiece con un chasquido.
        att = int(SAMPLE_RATE * attack_ms / 1000)
        if att > 0 and att < n:
            env[:att] *= np.linspace(0, 1, att, dtype=np.float32)
        return (s * env).astype(np.float32)

    def _noise(self, ms, decay=12):
        """Genera ruido blanco con caída exponencial. Base de golpes y bombas."""
        n = int(SAMPLE_RATE * ms / 1000)
        t = np.linspace(0, ms/1000, n, False, dtype=np.float32)
        s = (np.random.random(n).astype(np.float32) * 2 - 1)  # ruido en [-1,1]
        env = np.exp(-decay*t)
        return s * env

    def _slide(self, f0, f1, ms, decay=4, wave='sine'):
        """Tono cuya frecuencia se desliza linealmente de f0 a f1.

        Se usa para el 'oink' del cerdo (sube), el 'muu' de la vaca (baja),
        el kikiriki del gallo y el sub-bajo de la explosión."""
        n = int(SAMPLE_RATE * ms / 1000)
        t = np.linspace(0, ms/1000, n, False, dtype=np.float32)
        # Frecuencia instantánea en cada muestra (rampa lineal f0 -> f1).
        freq_t = f0 + (f1 - f0) * (t/(ms/1000))
        # La fase es la integral de la frecuencia: cumsum aproxima esa integral.
        phase = 2*np.pi*np.cumsum(freq_t)/SAMPLE_RATE
        if wave == 'square':
            s = np.sign(np.sin(phase))
        else:
            s = np.sin(phase)
        env = np.exp(-decay*t)
        return s * env

    def _mix(self, *arrays):
        """Suma varios arrays mono que pueden tener longitudes distintas
        (paddea con ceros al final)."""
        n = max(len(a) for a in arrays)
        out = np.zeros(n, dtype=np.float32)
        for a in arrays:
            out[:len(a)] += a
        return out

    def _to_sound(self, mono, volume=1.0):
        """Convierte un array mono de float a un objeto pygame.Sound reproducible.

        `volume` es el peso RELATIVO de este sonido en la mezcla (para que unos
        suenen más fuertes que otros). El volumen global del jugador se aplica
        aparte, en el momento de reproducir."""
        # Normalizar: dividir por el pico para que el sonido use todo el rango.
        peak = float(np.max(np.abs(mono))) if len(mono) else 1.0
        if peak > 0:
            mono = mono / peak
        mono = mono * volume
        mono = np.clip(mono, -1.0, 1.0)            # recortar por seguridad
        arr = (mono * 32767).astype(np.int16)      # a entero de 16 bits
        stereo = np.column_stack([arr, arr])       # duplicar canal: L = R
        return pg.sndarray.make_sound(stereo)

    # ---------- construcción de todos los sonidos ----------
    def _build_all(self):
        """Sintetiza los 10 efectos del juego y los guarda en self.sounds.
        Se llama una sola vez al iniciar. Cada sonido combina ondas básicas."""
        # click corto: dos tonos agudos breves (feedback al pulsar botones)
        click = self._mix(
            self._tone(1100, 50, decay=40, wave='sine'),
            0.4 * self._tone(2200, 30, decay=60),
        )
        self.sounds['click'] = self._to_sound(click, 0.6)

        # error: dos tonos graves disonantes seguidos (acción no permitida)
        err = np.concatenate([
            self._tone(220, 100, decay=10, wave='square'),
            self._tone(180, 140, decay=10, wave='square'),
        ])
        self.sounds['error'] = self._to_sound(err, 0.5)

        # push: golpe sordo al empujar una caja (ruido + tono grave)
        push = self._mix(
            self._noise(80, decay=22),
            0.6 * self._tone(80, 120, decay=18),
        )
        self.sounds['push'] = self._to_sound(push, 0.7)

        # explosion: detonación de la bomba (4 capas superpuestas)
        boom = self._mix(
            self._noise(600, decay=4),                          # cuerpo del bum
            1.2 * self._slide(120, 40, 500, decay=3, wave='saw'),  # sub-bajo
            0.8 * self._noise(80, decay=30),                    # chasquido inicial
            0.4 * self._tone(60, 700, decay=2.5),               # cola grave
        )
        self.sounds['explosion'] = self._to_sound(boom, 0.95)

        # step (oink corto): sonido del cerdo al dar un paso
        oink = self._mix(
            self._slide(220, 320, 70, decay=10, wave='saw'),
            0.5 * self._slide(440, 640, 70, decay=12, wave='saw'),
            0.3 * self._noise(70, decay=18),
        )
        self.sounds['step'] = self._to_sound(oink, 0.45)

        # oink completo (más expresivo)
        oink2 = np.concatenate([
            self._slide(180, 280, 90, decay=8, wave='saw'),
            self._slide(280, 200, 110, decay=8, wave='saw'),
        ])
        oink2 = self._mix(oink2, 0.3 * self._noise(200, decay=10))
        self.sounds['oink'] = self._to_sound(oink2, 0.6)

        # muu de vaca
        muu = self._mix(
            self._slide(110, 95, 700, decay=2, wave='saw'),
            0.5 * self._slide(220, 190, 700, decay=2.5),
        )
        self.sounds['muu'] = self._to_sound(muu, 0.55)

        # kikiriki de gallo
        kiki = np.concatenate([
            self._slide(500, 700, 90, decay=6),
            self._silence(40),
            self._slide(700, 900, 70, decay=8),
            self._silence(30),
            self._slide(900, 600, 200, decay=4),
        ])
        self.sounds['kikiriki'] = self._to_sound(kiki, 0.5)

        # win level: do-mi-sol-do ascendente
        notes = [(523, 120), (659, 120), (784, 120), (1047, 280)]
        wl = np.concatenate([self._tone(f, ms, decay=4) for f, ms in notes])
        self.sounds['win_level'] = self._to_sound(wl, 0.55)

        # win game: fanfarria
        parts = []
        for f, ms in [(523, 150), (659, 150), (784, 150), (1047, 350),
                      (784, 150), (1047, 500)]:
            parts.append(self._tone(f, ms, decay=3.5))
            parts.append(self._silence(20))
        chord = self._mix(
            self._tone(523, 800, decay=2),
            self._tone(659, 800, decay=2),
            self._tone(784, 800, decay=2),
        ) / 3
        parts.append(chord)
        self.sounds['win_game'] = self._to_sound(np.concatenate(parts), 0.55)

        # ambient granja: loop con animales aleatorios + ruido suave
        amb_ms = 6000
        n = int(SAMPLE_RATE * amb_ms / 1000)
        bed = self._noise(amb_ms, decay=0) * 0.04
        win = 200
        bed = np.convolve(bed, np.ones(win)/win, mode='same').astype(np.float32)
        track = bed.copy()
        rng = random.Random(123)
        for _ in range(5):
            pos_ms = rng.randint(200, amb_ms - 1000)
            kind = rng.choice(['oink', 'muu', 'kikiriki'])
            if kind == 'oink':
                snd = self._slide(180+rng.randint(-20,20),
                                  260+rng.randint(-20,20),
                                  150, decay=8, wave='saw')
            elif kind == 'muu':
                snd = self._slide(100, 88, 600, decay=2.5, wave='saw')
            else:
                snd = self._slide(550, 850, 150, decay=6)
            start = int(SAMPLE_RATE * pos_ms / 1000)
            end = start + len(snd)
            if end < len(track):
                track[start:end] += snd * 0.35
        self.sounds['farm_ambient'] = self._to_sound(track, 0.5)

    # ---------- API pública ----------
    # Estos son los únicos métodos que el resto del juego debe llamar.

    def play(self, name):
        """Reproduce una vez el sonido indicado. Respeta los conmutadores de
        música/efectos y el volumen correspondiente. No-op si el audio falló."""
        if not self.ok:
            return
        is_music = name in self._music_names
        # Saltar si la categoría correspondiente está desactivada.
        if is_music and not self.music_enabled:
            return
        if not is_music and not self.sfx_enabled:
            return
        s = self.sounds.get(name)
        if s is None:
            return
        vol = self.music_volume if is_music else self.sfx_volume
        s.set_volume(vol)
        s.play()

    def play_loop(self, name):
        """Reproduce un sonido en bucle infinito (la música de ambiente).
        Si ya hay un bucle sonando, no hace nada (evita solaparlos)."""
        if not self.ok:
            return
        if name in self._music_names and not self.music_enabled:
            return
        s = self.sounds.get(name)
        if s is None:
            return
        # Si el canal de bucle ya está ocupado, no arrancamos otro.
        if self.loop_channel is not None and self.loop_channel.get_busy():
            return
        ch = pg.mixer.find_channel()      # buscar un canal libre
        if ch is None:
            ch = pg.mixer.Channel(0)      # si no hay, usar el 0 forzosamente
        self.loop_channel = ch
        ch.set_volume(self.music_volume)
        ch.play(s, loops=-1)              # loops=-1 => repetir para siempre

    def stop_loop(self):
        """Detiene la música en bucle (al salir del menú principal)."""
        if self.loop_channel is not None:
            self.loop_channel.stop()
            self.loop_channel = None

    def set_music_enabled(self, enabled):
        """Activa/desactiva la música. Al desactivar, corta el bucle actual."""
        self.music_enabled = bool(enabled)
        if not self.music_enabled:
            self.stop_loop()

    def set_sfx_enabled(self, enabled):
        """Activa/desactiva los efectos de sonido."""
        self.sfx_enabled = bool(enabled)

    def set_music_volume(self, vol):
        """Fija el volumen de música (0.0–1.0) y lo aplica al bucle en curso."""
        self.music_volume = max(0.0, min(1.0, float(vol)))
        if self.loop_channel is not None:
            self.loop_channel.set_volume(self.music_volume)

    def set_sfx_volume(self, vol):
        """Fija el volumen de los efectos de sonido (0.0–1.0)."""
        self.sfx_volume = max(0.0, min(1.0, float(vol)))


# ── Solver BFS para Sokoban ────────────────────────────────────────────────────
def solve_level(level_grid, max_states=1_000_000):
    """Resuelve un nivel de Sokoban con búsqueda en anchura (BFS).

    Como BFS explora los estados por capas (primero los alcanzables en 1 paso,
    luego en 2, etc.), la PRIMERA solución que encuentra usa el número mínimo
    de movimientos posible. Ese número es el "óptimo" del nivel.

    Parámetros:
      level_grid : matriz de strings con el nivel (ver get_levels para los códigos)
      max_states : tope de estados a explorar; evita que se cuelgue en niveles
                   imposibles o demasiado complejos.

    Devuelve:
      int  -> número mínimo de movimientos del jugador
      0    -> el nivel ya está resuelto de inicio
      None -> no hay solución, o se superó max_states sin encontrarla

    Un "estado" del juego es la pareja (posición del jugador, conjunto de cajas).
    Las paredes no forman parte del estado porque nunca cambian.
    """
    h = len(level_grid)                       # alto del tablero
    w = len(level_grid[0]) if h else 0        # ancho del tablero
    if not h:
        return None

    # Recorremos el tablero una vez para extraer los elementos del nivel.
    player = None                             # posición inicial del jugador
    boxes, targets, blocked = set(), set(), set()

    for y in range(h):
        for x in range(w):
            cell = level_grid[y][x]
            if cell in ('w', 'g'):            # pared o galaxia: intransitable
                blocked.add((x, y))
            if cell == 'a':                   # jugador sobre suelo
                player = (x, y)
            elif cell == 'aot':               # jugador sobre objetivo
                player = (x, y); targets.add((x, y))
            elif cell == 'b':                 # caja sobre suelo
                boxes.add((x, y))
            elif cell == 'bot':               # caja ya sobre objetivo
                boxes.add((x, y)); targets.add((x, y))
            elif cell == 'o':                 # objetivo vacío
                targets.add((x, y))

    if player is None:
        return None                          # nivel inválido: sin jugador
    # Caso trivial: si las cajas ya coinciden con los objetivos, óptimo = 0.
    if boxes == targets and len(boxes) == len(targets):
        return 0

    # El estado inicial. `frozenset` permite usar el conjunto de cajas como
    # clave en `visited` (los set normales no son hashables).
    initial = (player, frozenset(boxes))
    visited = {initial}                       # estados ya vistos (no repetir)
    # La cola guarda tripletas (posición jugador, cajas, nº de movimientos).
    queue = deque([(player, frozenset(boxes), 0)])

    while queue:
        # Corte de seguridad: si exploramos demasiados estados, abandonamos.
        if len(visited) > max_states:
            return None
        (px, py), bxs, moves = queue.popleft()   # sacar el más antiguo (FIFO)

        # Probar los 4 movimientos: arriba, abajo, izquierda, derecha.
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = px + dx, py + dy            # casilla destino del jugador
            # Fuera del tablero o contra una pared: movimiento inválido.
            if not (0 <= nx < w and 0 <= ny < h) or (nx, ny) in blocked:
                continue

            new_boxes = bxs
            # Si en el destino hay una caja, hay que intentar EMPUJARLA.
            if (nx, ny) in bxs:
                bx2, by2 = nx + dx, ny + dy      # casilla tras la caja
                # La caja no puede salir del tablero...
                if not (0 <= bx2 < w and 0 <= by2 < h):
                    continue
                # ...ni chocar con pared u otra caja.
                if (bx2, by2) in blocked or (bx2, by2) in bxs:
                    continue
                # Empuje válido: la caja se mueve una casilla.
                new_boxes = frozenset((bxs - {(nx, ny)}) | {(bx2, by2)})

            state = ((nx, ny), new_boxes)
            if state in visited:                 # ya explorado: saltar
                continue
            visited.add(state)

            nm = moves + 1
            # ¿Ganamos? Todas las cajas sobre todos los objetivos.
            if new_boxes == targets and len(new_boxes) == len(targets):
                return nm
            queue.append(((nx, ny), new_boxes, nm))
    return None   # cola vacía sin solución: nivel irresoluble


# ── Generadores de skins (personajes) ─────────────────────────────────────────
def _surf(size):
    """Crea una superficie cuadrada vacía con canal alfa (transparencia)."""
    return pg.Surface((size, size), pg.SRCALPHA)


# ── Generadores de skins (personajes) ─────────────────────────────────────────
# Cada función make_XXX(size) DIBUJA un personaje de forma procedural (con
# primitivas de pygame: círculos, polígonos, líneas) y devuelve una superficie
# cuadrada de lado `size` con transparencia. No se usan archivos de imagen.
# Así el juego no depende de assets externos para los personajes vectoriales.
def _surf(size):
    """Crea una superficie cuadrada vacía con canal alfa (transparencia)."""
    return pg.Surface((size, size), pg.SRCALPHA)


def make_pig(size):
    """Dibuja el cerdo vectorial: cara rosa, hocico, ojos y orejas triangulares.
    (Nota: si existe el archivo pig_skin.png, el juego usa esa imagen en su
    lugar; este dibujo es el respaldo.)"""
    s = _surf(size)
    pg.draw.circle(s, (240, 160, 160), (size//2, size//2), int(size*0.36))
    pg.draw.ellipse(s, (220, 130, 130),
                    (size//2 - size//8, size//2, size//4, size//6))
    pg.draw.circle(s, (50, 30, 20), (size//2 - size//6, size//2 - size//8), size//14)
    pg.draw.circle(s, (50, 30, 20), (size//2 + size//6, size//2 - size//8), size//14)
    pg.draw.polygon(s, (240, 160, 160), [
        (size//2 - size//3, size//4),
        (size//2 - size//5, size//6),
        (size//2 - size//8, size//4)])
    pg.draw.polygon(s, (240, 160, 160), [
        (size//2 + size//3, size//4),
        (size//2 + size//5, size//6),
        (size//2 + size//8, size//4)])
    return s


def make_robot(size):
    """Dibuja el robot: cuerpo rectangular, cabeza con ojos cian y antena."""
    s = _surf(size)
    pg.draw.rect(s, (160, 165, 185), (size//4, size//3, size//2, size//2),
                 border_radius=size//12)
    pg.draw.rect(s, (90, 95, 120), (size//4, size//3, size//2, size//2),
                 3, border_radius=size//12)
    pg.draw.rect(s, (200, 205, 225),
                 (size//3, size//5, size//3, size//4), border_radius=size//16)
    pg.draw.rect(s, (90, 95, 120),
                 (size//3, size//5, size//3, size//4), 2, border_radius=size//16)
    pg.draw.circle(s, (80, 220, 255),
                   (size//2 - size//12, size//3 - size//40), size//22)
    pg.draw.circle(s, (80, 220, 255),
                   (size//2 + size//12, size//3 - size//40), size//22)
    pg.draw.line(s, (90, 95, 120), (size//2, size//5), (size//2, size//10), 3)
    pg.draw.circle(s, (255, 90, 90), (size//2, size//10), size//22)
    return s


def make_ninja(size):
    """Dibuja el ninja: cabeza oscura con banda roja y ojos visibles."""
    s = _surf(size)
    pg.draw.circle(s, (45, 45, 60), (size//2, size//2), int(size*0.36))
    pg.draw.rect(s, (200, 60, 60),
                 (size//2 - size//3, size//2 - size//14, int(size*0.66), size//7))
    eye_y = size//2 - size//10
    pg.draw.line(s, (240, 240, 240),
                 (size//2 - size//5, eye_y), (size//2 - size//12, eye_y), 3)
    pg.draw.line(s, (240, 240, 240),
                 (size//2 + size//12, eye_y), (size//2 + size//5, eye_y), 3)
    pg.draw.line(s, (200, 60, 60),
                 (size//2 - size//3, size//2 + size//9),
                 (size//2 - int(size*0.45), size//2 + size//5), 3)
    pg.draw.line(s, (200, 60, 60),
                 (size//2 - int(size*0.45), size//2 + size//5),
                 (size//2 - size//4, size//2 + int(size*0.32)), 3)
    return s


def make_astronaut(size):
    """Dibuja el astronauta: casco blanco con visera azul reflectante."""
    s = _surf(size)
    pg.draw.circle(s, (235, 235, 245), (size//2, size//2), int(size*0.36))
    pg.draw.circle(s, (180, 185, 200), (size//2, size//2), int(size*0.36), 3)
    pg.draw.ellipse(s, (90, 140, 220),
                    (size//2 - size//4, size//2 - size//6, size//2, size//3))
    pg.draw.ellipse(s, (50, 90, 170),
                    (size//2 - size//4, size//2 - size//6, size//2, size//3), 2)
    pg.draw.ellipse(s, (180, 220, 255),
                    (size//2 - size//6, size//2 - size//10, size//4, size//8))
    pg.draw.line(s, (180, 185, 200),
                 (size//2 - int(size*0.36), size//2 - size//12),
                 (size//2 - int(size*0.20), size//2 - size//12), 3)
    return s


def make_dragon(size):
    """Dibuja el dragón: cabeza verde con cuernos rojos, ojos amarillos y morros."""
    s = _surf(size)
    pg.draw.circle(s, (90, 180, 100), (size//2, size//2), int(size*0.36))
    pg.draw.polygon(s, (200, 60, 60), [
        (size//2 - size//4, size//4),
        (size//2 - size//5, size//8),
        (size//2 - size//8, size//4)])
    pg.draw.polygon(s, (200, 60, 60), [
        (size//2 + size//4, size//4),
        (size//2 + size//5, size//8),
        (size//2 + size//8, size//4)])
    pg.draw.circle(s, (255, 240, 80),
                   (size//2 - size//7, size//2 - size//10), size//14)
    pg.draw.circle(s, (255, 240, 80),
                   (size//2 + size//7, size//2 - size//10), size//14)
    pg.draw.circle(s, (40, 20, 20),
                   (size//2 - size//7, size//2 - size//10), size//28)
    pg.draw.circle(s, (40, 20, 20),
                   (size//2 + size//7, size//2 - size//10), size//28)
    pg.draw.circle(s, (60, 130, 70),
                   (size//2 - size//8, size//2 + size//6), size//22)
    pg.draw.circle(s, (60, 130, 70),
                   (size//2 + size//8, size//2 + size//6), size//22)
    return s


# ── Generadores de cajas ──────────────────────────────────────────────────────
# Igual que los personajes: cada función dibuja una caja de un material distinto.
def make_wooden_box(size):
    """Caja de madera: la básica, marrón con aspas en diagonal."""
    s = _surf(size)
    pg.draw.rect(s, (190, 140, 60), (3, 3, size-6, size-6))
    pg.draw.rect(s, (140, 100, 30), (3, 3, size-6, size-6), 4)
    pg.draw.line(s, (140, 100, 30), (5, 5), (size-5, size-5), 2)
    pg.draw.line(s, (140, 100, 30), (size-5, 5), (5, size-5), 2)
    return s


def make_metal_box(size):
    """Caja metálica: gris con remaches en las esquinas y refuerzo en cruz."""
    s = _surf(size)
    pg.draw.rect(s, (150, 155, 165), (3, 3, size-6, size-6))
    pg.draw.rect(s, (80, 85, 95), (3, 3, size-6, size-6), 4)
    for cx, cy in [(10, 10), (size-10, 10), (10, size-10), (size-10, size-10)]:
        pg.draw.circle(s, (60, 65, 75), (cx, cy), 3)
    pg.draw.line(s, (80, 85, 95), (size//2, 5), (size//2, size-5), 2)
    pg.draw.line(s, (80, 85, 95), (5, size//2), (size-5, size//2), 2)
    return s


def make_crystal_box(size):
    """Caja de cristal: azul translúcida con un reflejo triangular."""
    s = _surf(size)
    pg.draw.rect(s, (140, 200, 230, 200), (3, 3, size-6, size-6))
    pg.draw.rect(s, (50, 140, 200), (3, 3, size-6, size-6), 4)
    pg.draw.polygon(s, (220, 240, 255, 180), [
        (size//4, size//4), (size//2, size//4),
        (size//4, size//2)])
    pg.draw.line(s, (50, 140, 200), (3, 3), (size-3, size-3), 2)
    pg.draw.line(s, (50, 140, 200), (size-3, 3), (3, size-3), 2)
    return s


def make_gold_box(size):
    """Caja dorada: amarilla con un destello en forma de estrella de 8 puntas."""
    s = _surf(size)
    pg.draw.rect(s, (240, 200, 70), (3, 3, size-6, size-6))
    pg.draw.rect(s, (180, 130, 30), (3, 3, size-6, size-6), 4)
    pg.draw.line(s, (180, 130, 30), (5, 5), (size-5, size-5), 2)
    pg.draw.line(s, (180, 130, 30), (size-5, 5), (5, size-5), 2)
    pg.draw.polygon(s, (255, 240, 180), [
        (size//2, size//4),
        (size//2 + size//12, size//2 - size//12),
        (size*3//4, size//2),
        (size//2 + size//12, size//2 + size//12),
        (size//2, size*3//4),
        (size//2 - size//12, size//2 + size//12),
        (size//4, size//2),
        (size//2 - size//12, size//2 - size//12)])
    return s


def make_magic_box(size):
    """Caja mágica: morada con estrellitas dispersas y un destello central.
    Usa random para la posición de las estrellas (varía en cada generación)."""
    s = _surf(size)
    pg.draw.rect(s, (110, 60, 170), (3, 3, size-6, size-6))
    pg.draw.rect(s, (60, 20, 100), (3, 3, size-6, size-6), 4)
    for _ in range(6):
        x = random.randint(8, size-8)
        y = random.randint(8, size-8)
        pg.draw.circle(s, (255, 240, 180), (x, y), 2)
    cx, cy = size//2, size//2
    r = size//5
    pg.draw.polygon(s, (255, 230, 120), [
        (cx, cy - r), (cx + r//3, cy - r//3),
        (cx + r, cy), (cx + r//3, cy + r//3),
        (cx, cy + r), (cx - r//3, cy + r//3),
        (cx - r, cy), (cx - r//3, cy - r//3)])
    return s


def add_target_overlay(surf, size):
    """Devuelve una COPIA de `surf` con un borde verde, para indicar que el
    actor (jugador o caja) está colocado sobre una casilla-objetivo."""
    s = surf.copy()
    pg.draw.rect(s, (80, 220, 100), (1, 1, size-2, size-2), 4)
    return s


# ── Catálogos de skins ────────────────────────────────────────────────────────
# Diccionarios que asocian una clave interna con: nombre visible, precio en
# monedas y la función generadora. Los usa la tienda y el dibujado del tablero.
CHARACTER_SKINS = {
    'pig':       {'name': 'Cerdito',    'price':   0, 'maker': make_pig},
    'robot':     {'name': 'Robot',      'price': 100, 'maker': make_robot},
    'ninja':     {'name': 'Ninja',      'price': 200, 'maker': make_ninja},
    'astronaut': {'name': 'Astronauta', 'price': 300, 'maker': make_astronaut},
    'dragon':    {'name': 'Dragón',     'price': 500, 'maker': make_dragon},
}

BOX_SKINS = {
    'wooden':  {'name': 'Madera',  'price':   0, 'maker': make_wooden_box},
    'metal':   {'name': 'Metal',   'price': 100, 'maker': make_metal_box},
    'crystal': {'name': 'Cristal', 'price': 200, 'maker': make_crystal_box},
    'gold':    {'name': 'Dorada',  'price': 400, 'maker': make_gold_box},
    'magic':   {'name': 'Mágica',  'price': 600, 'maker': make_magic_box},
}


# ── Items consumibles ─────────────────────────────────────────────────────────
# Iconos de los dos items comprables. A diferencia de skins, los items se gastan.
def make_bomb_icon(size):
    """Dibuja el icono de la bomba: esfera negra con mecha y chispa."""
    s = pg.Surface((size, size), pg.SRCALPHA)
    # cuerpo negro redondo
    pg.draw.circle(s, (30, 30, 35), (size//2, size*3//5), int(size*0.32))
    pg.draw.circle(s, (60, 60, 70), (size//2, size*3//5), int(size*0.32), 2)
    # brillo
    pg.draw.circle(s, (120, 120, 130),
                   (size//2 - size//8, size*3//5 - size//8), size//12)
    # mecha
    pg.draw.line(s, (140, 100, 60),
                 (size//2 + size//8, size//2 - size//12),
                 (size*3//4, size//4), 3)
    # chispa
    for ang in range(0, 360, 60):
        import math
        x = size*3//4 + int(math.cos(math.radians(ang)) * 8)
        y = size//4 + int(math.sin(math.radians(ang)) * 8)
        pg.draw.line(s, (255, 200, 80),
                     (size*3//4, size//4), (x, y), 2)
    pg.draw.circle(s, (255, 230, 100), (size*3//4, size//4), 5)
    return s


def make_wand_icon(size):
    """Dibuja el icono del pase mágico: una varita dorada con destellos."""
    s = pg.Surface((size, size), pg.SRCALPHA)
    # varita inclinada
    import math
    cx, cy = size//2, size//2
    a = math.radians(-30)
    x1, y1 = int(cx - 22 * math.cos(a)), int(cy - 22 * math.sin(a))
    x2, y2 = int(cx + 22 * math.cos(a)), int(cy + 22 * math.sin(a))
    pg.draw.line(s, (110, 70, 30), (x1, y1), (x2, y2), 5)
    # punta dorada
    pg.draw.circle(s, (255, 230, 100), (x2, y2), 6)
    pg.draw.circle(s, (200, 160, 50), (x2, y2), 6, 2)
    # estrellas alrededor
    for off in (-12, 8, -4):
        sx = x2 + int(math.cos(a + math.pi/2) * off)
        sy = y2 + int(math.sin(a + math.pi/2) * off)
        pg.draw.line(s, (255, 240, 180), (sx-3, sy), (sx+3, sy), 1)
        pg.draw.line(s, (255, 240, 180), (sx, sy-3), (sx, sy+3), 1)
    return s


# Catálogo de items: clave -> nombre, precio, generador de icono y descripción.
ITEMS = {
    'bomb':  {'name': 'Bomba',       'price':  30,
              'maker': make_bomb_icon,
              'desc': 'Destruye una pared'},
    'magic': {'name': 'Pase mágico', 'price':  60,
              'maker': make_wand_icon,
              'desc': 'Coloca una caja en un objetivo'},
}


# ── Gestor de datos persistentes ──────────────────────────────────────────────
class GameData:
    """Encapsula TODO lo que se guarda en disco entre sesiones.

    Trabaja sobre un único diccionario `self.data` que se serializa como JSON.
    Estructura de self.data:
      players    : { nombre: {nivel, monedas, skins, records, items, ...} }
      min_moves  : { "nivel": óptimo }  -> caché de resultados del solver
      level_best : { "nivel": [ {name, moves, pushes, time_ms}, ... ] }
      settings   : { music_enabled, sfx_enabled, music_volume, sfx_volume }

    Cada método que modifica datos llama a save() para persistir de inmediato.
    """

    def __init__(self, path=SAVE_FILE):
        """Crea el gestor y carga los datos del disco (si el archivo existe)."""
        self.path = path
        # Estructura por defecto (se sobrescribe en load() si hay archivo).
        self.data = {'players': {}, 'min_moves': {}, 'level_best': {},
                     'settings': self._default_settings()}
        self.load()

    @staticmethod
    def _default_settings():
        """Ajustes de fábrica: audio activado a volumen medio."""
        return {
            'music_enabled': True,
            'sfx_enabled': True,
            'music_volume': 0.5,
            'sfx_volume': 0.6,
        }

    def get_setting(self, key):
        """Lee un ajuste; si no existe, devuelve el valor por defecto."""
        return self.data.get('settings', {}).get(key,
            self._default_settings().get(key))

    def set_setting(self, key, value):
        """Cambia un ajuste y guarda en disco."""
        self.data.setdefault('settings', self._default_settings())
        self.data['settings'][key] = value
        self.save()

    def load(self):
        """Carga self.data desde el JSON. Si el archivo está corrupto o no
        existe, deja la estructura por defecto. Hace 'merge' de settings para
        no perder claves nuevas al actualizar el juego."""
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
                # setdefault garantiza que las claves principales existen
                # aunque el archivo sea de una versión antigua del juego.
                self.data.setdefault('players', {})
                self.data.setdefault('min_moves', {})
                self.data.setdefault('level_best', {})
                # Combinar settings guardados con los por defecto.
                saved_settings = self.data.get('settings', {})
                merged = self._default_settings()
                merged.update(saved_settings)
                self.data['settings'] = merged
            except Exception:
                # Archivo ilegible: empezar de cero sin reventar el juego.
                self.data = {'players': {}, 'min_moves': {}, 'level_best': {},
                             'settings': self._default_settings()}

    def save(self):
        """Escribe self.data al archivo JSON (con indentación legible)."""
        try:
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print('Error al guardar:', e)

    def _norm(self, name):
        """Normaliza un nombre para comparar: sin espacios extra y en minúsculas.
        Así 'Ana' y ' ana ' se consideran el mismo jugador."""
        return name.strip().lower()

    def player_exists(self, name):
        """True si ya hay un jugador con ese nombre (ignorando may/min)."""
        target = self._norm(name)
        return any(self._norm(k) == target for k in self.data['players'])

    def find_canonical(self, name):
        """Devuelve el nombre tal y como está guardado realmente, o None.
        Necesario porque el usuario puede escribir 'ANA' y estar guardado 'Ana'."""
        target = self._norm(name)
        for k in self.data['players']:
            if self._norm(k) == target:
                return k
        return None

    def add_player(self, name):
        """Crea un jugador nuevo con todos los valores iniciales y guarda."""
        name = name.strip()
        self.data['players'][name] = {
            'level': 1,                  # nivel desbloqueado más alto + 1
            'max_completed': 0,          # nivel más alto completado
            'coins': 0,                  # monedas acumuladas
            'character': 'pig',          # skin de personaje equipada
            'box_skin': 'wooden',        # skin de caja equipada
            'unlocked_chars': ['pig'],   # personajes comprados
            'unlocked_boxes': ['wooden'],# cajas compradas
            'optimal_levels': [],        # niveles superados con ruta óptima
            'records': {},               # mejor marca personal por nivel
            'items': {'bomb': 0, 'magic': 0},  # inventario de consumibles
        }
        self.save()

    def add_item(self, name, item_key, count=1):
        """Añade `count` unidades de un item al inventario del jugador."""
        player = self.data['players'].get(name)
        if player is None:
            return False
        items = dict(player.get('items', {}))
        items[item_key] = items.get(item_key, 0) + count
        player['items'] = items
        self.save()
        return True

    def use_item(self, name, item_key):
        """Consume una unidad de un item del inventario del jugador.
        Devuelve False si no le quedaba ninguno."""
        player = self.data['players'].get(name)
        if player is None:
            return False
        items = dict(player.get('items', {}))
        if items.get(item_key, 0) <= 0:
            return False
        items[item_key] -= 1
        player['items'] = items
        self.save()
        return True

    def get_item_count(self, name, item_key):
        """Cuántas unidades de un item tiene el jugador (0 si no tiene)."""
        player = self.data['players'].get(name)
        if player is None:
            return 0
        return player.get('items', {}).get(item_key, 0)

    def delete_player(self, name):
        """Borra un jugador del archivo de guardado. Devuelve True si existía."""
        canonical = self.find_canonical(name)
        if canonical is not None:
            del self.data['players'][canonical]
            self.save()
            return True
        return False

    def get_player(self, name):
        """Devuelve el diccionario de un jugador (o None si no existe)."""
        return self.data['players'].get(name)

    def update_player(self, name, **fields):
        """Actualiza uno o varios campos de un jugador y guarda.
        Ejemplo: update_player('Ana', coins=120, level=5)."""
        if name in self.data['players']:
            self.data['players'][name].update(fields)
            self.save()

    def ranking(self):
        """Devuelve la lista de jugadores ORDENADA para el ranking global:
        primero por niveles completados (desc), luego por monedas (desc) y
        por último alfabéticamente. Devuelve pares (nombre, datos)."""
        items = list(self.data['players'].items())
        items.sort(key=lambda kv: (-kv[1].get('max_completed', 0),
                                   -kv[1].get('coins', 0),
                                    kv[0].lower()))
        return items

    def get_min_moves(self, level):
        """Óptimo cacheado de un nivel (el solver lo calcula una vez)."""
        return self.data['min_moves'].get(str(level))

    def set_min_moves(self, level, value):
        """Guarda en caché el óptimo de un nivel para no recalcularlo."""
        self.data['min_moves'][str(level)] = value
        self.save()

    def get_level_top(self, level, limit=5):
        """Devuelve la tabla de mejores marcas de un nivel (ya ordenada)."""
        return list(self.data['level_best'].get(str(level), []))[:limit]

    def submit_level_run(self, name, level, moves, pushes, time_ms):
        """Registra una partida superada en la tabla de récords del nivel.

        La tabla se ordena por (movimientos, tiempo, empujones) ascendente, así
        que la mejor marca queda primera. Solo se guarda UN registro por jugador
        (el mejor) y un máximo de 10 entradas por nivel.

        Devuelve: (posición_en_la_tabla, batió_su_récord_de_movs,
                   batió_su_récord_de_tiempo)
        """
        key = str(level)
        board = list(self.data['level_best'].get(key, []))

        # Separar el registro previo de este jugador (si lo había) del resto.
        prev = None
        new_board = []
        for r in board:
            if r.get('name') == name:
                prev = r
            else:
                new_board.append(r)
        board = new_board

        # Solo se acepta la marca nueva si mejora la anterior del jugador.
        accept = True
        if prev is not None:
            # Comparación lexicográfica de tuplas: primero movs, luego tiempo...
            if (moves, time_ms, pushes) >= (prev['moves'],
                                            prev.get('time_ms', 10**9),
                                            prev.get('pushes', 10**9)):
                accept = False

        # ¿Batió sus propios récords personales? (para premiar con monedas)
        beat_self_moves = prev is None or moves < prev['moves']
        beat_self_time = prev is None or time_ms < prev.get('time_ms', 10**9)

        entry = {'name': name, 'moves': moves, 'pushes': pushes,
                 'time_ms': time_ms}

        if accept:
            board.append(entry)            # entra la marca nueva
        elif prev is not None:
            board.append(prev)             # se conserva la anterior (mejor)

        # Reordenar y recortar a las 10 mejores.
        board.sort(key=lambda r: (r['moves'],
                                  r.get('time_ms', 10**9),
                                  r.get('pushes', 10**9)))
        board = board[:10]
        self.data['level_best'][key] = board

        # Actualizar también el récord personal del jugador (records[nivel]).
        player = self.data['players'].get(name)
        if player is not None:
            recs = dict(player.get('records', {}))
            current = recs.get(key)
            if current is None or (moves, time_ms) < (current['moves'],
                                                      current.get('time_ms',
                                                                  10**9)):
                recs[key] = {'moves': moves, 'time_ms': time_ms,
                             'pushes': pushes}
                player['records'] = recs

        self.save()

        # Posición final del jugador en la tabla (1 = primero), o None.
        position = next((i+1 for i, r in enumerate(board)
                         if r.get('name') == name), None)
        return position, beat_self_moves, beat_self_time


# ── Definición de los 20 niveles ──────────────────────────────────────────────
def get_levels():
    """Devuelve un diccionario {número_de_nivel: matriz_9x9}.

    Cada nivel es una lista de 9 filas; cada fila es una lista de 9 strings.
    Códigos de celda usados en las matrices:
       'g' = galaxia  -> exterior del recinto (intransitable, fondo de estrellas)
       'w' = wall     -> pared (intransitable)
       'f' = floor    -> suelo libre
       'a' = avatar   -> posición inicial del jugador
       'b' = box      -> caja
       'o' = objetivo -> casilla destino donde debe quedar una caja
       'bot' = box on target  -> caja que ya empieza sobre un objetivo
       'aot' = avatar on target -> jugador que empieza sobre un objetivo

    Los nombres cortos g/w/f/a/b/o se asignan abajo para que las matrices
    queden compactas y legibles como un "dibujo" del nivel.

    REGLA DE DISEÑO: en cada nivel el nº de cajas debe ser igual al nº de
    objetivos, y el nivel debe ser resoluble (verificado con solve_level).
    La dificultad crece con el número de nivel (óptimos de 6 a 35 movimientos).
    """
    g, w, f, a, b, o = 'g', 'w', 'f', 'a', 'b', 'o'
    return {
        1: [
            [g,g,g,g,g,g,g,g,g],
            [g,w,w,w,w,w,g,g,g],
            [g,w,f,f,f,w,g,g,g],
            [g,w,f,b,f,w,g,g,g],
            [g,w,f,a,o,w,g,g,g],
            [g,w,w,w,w,w,g,g,g],
            [g,g,g,g,g,g,g,g,g],
            [g,g,g,g,g,g,g,g,g],
            [g,g,g,g,g,g,g,g,g],
        ],
        2: [
            [g,g,w,w,w,w,w,g,g],
            [g,g,w,f,f,f,w,g,g],
            [g,w,w,f,b,f,w,w,g],
            [g,w,o,f,a,f,f,w,g],
            [g,w,w,w,f,w,f,w,g],
            [g,g,g,w,f,w,f,w,g],
            [g,g,g,w,f,f,f,w,g],
            [g,g,g,w,w,w,w,w,g],
            [g,g,g,g,g,g,g,g,g],
        ],
        3: [
            [g,w,w,w,w,w,w,g,g],
            [g,w,f,f,f,f,w,g,g],
            [g,w,f,w,w,f,w,g,g],
            [g,w,f,b,f,f,w,g,g],
            [g,w,f,w,f,o,w,g,g],
            [g,w,a,f,f,f,w,g,g],
            [g,w,w,w,w,w,w,g,g],
            [g,g,g,g,g,g,g,g,g],
            [g,g,g,g,g,g,g,g,g],
        ],
        4: [
            [g,w,w,w,w,w,w,w,g],
            [g,w,f,f,f,f,f,w,g],
            [g,w,f,b,w,b,f,w,g],
            [g,w,f,f,a,f,f,w,g],
            [g,w,o,f,f,f,o,w,g],
            [g,w,f,f,f,f,f,w,g],
            [g,w,w,w,w,w,w,w,g],
            [g,g,g,g,g,g,g,g,g],
            [g,g,g,g,g,g,g,g,g],
        ],
        5: [
            [w,w,w,w,w,w,w,w,w],
            [w,f,f,f,o,f,f,f,w],
            [w,f,w,w,f,w,w,f,w],
            [w,f,b,f,b,f,f,f,w],
            [w,f,f,f,a,f,w,f,w],
            [w,f,w,f,f,f,w,f,w],
            [w,f,f,f,o,f,f,f,w],
            [w,w,w,w,w,w,w,w,w],
            [g,g,g,g,g,g,g,g,g],
        ],
        6: [
            [w,w,w,w,w,w,w,w,w],
            [w,f,f,f,o,f,f,f,w],
            [w,f,b,f,f,f,b,f,w],
            [w,f,f,w,f,w,f,f,w],
            [w,o,f,f,a,f,f,o,w],
            [w,f,f,w,f,w,f,f,w],
            [w,f,f,f,b,f,f,f,w],
            [w,f,f,f,f,f,f,f,w],
            [w,w,w,w,w,w,w,w,w],
        ],
        7: [
            [w,w,w,w,w,w,w,w,w],
            [w,f,f,f,f,f,f,f,w],
            [w,f,w,f,o,f,w,f,w],
            [w,f,f,b,f,b,f,f,w],
            [w,f,f,f,a,f,f,f,w],
            [w,f,f,f,b,f,f,f,w],
            [w,f,w,f,o,f,w,f,w],
            [w,f,o,f,f,f,f,f,w],
            [w,w,w,w,w,w,w,w,w],
        ],
        8: [
            [w,w,w,w,w,w,w,w,w],
            [w,o,f,f,f,f,f,o,w],
            [w,f,f,w,f,w,f,f,w],
            [w,w,f,b,f,b,f,w,w],
            [w,f,f,f,a,f,f,f,w],
            [w,w,f,f,b,f,f,w,w],
            [w,f,f,w,f,w,f,f,w],
            [w,f,f,f,o,f,f,f,w],
            [w,w,w,w,w,w,w,w,w],
        ],
        9: [
            [w,w,w,w,w,w,w,w,w],
            [w,o,f,f,f,f,f,o,w],
            [w,f,w,f,f,f,w,f,w],
            [w,f,b,f,f,f,b,f,w],
            [w,f,f,f,a,f,f,f,w],
            [w,f,f,f,b,f,f,f,w],
            [w,f,w,w,f,w,w,f,w],
            [w,f,f,f,o,f,f,f,w],
            [w,w,w,w,w,w,w,w,w],
        ],
        10: [
            [w,w,w,w,w,w,w,w,w],
            [w,o,f,f,f,f,f,o,w],
            [w,f,f,w,f,w,f,f,w],
            [w,f,b,f,w,f,b,f,w],
            [w,f,f,f,a,f,f,f,w],
            [w,f,f,f,b,f,f,f,w],
            [w,f,f,w,f,w,f,f,w],
            [w,f,f,f,o,f,f,f,w],
            [w,w,w,w,w,w,w,w,w],
        ],
        11: [
            [w,w,w,w,w,w,w,w,w],
            [w,o,f,f,f,f,f,o,w],
            [w,f,w,f,f,f,w,f,w],
            [w,f,b,f,f,f,b,f,w],
            [w,f,w,f,a,f,w,f,w],
            [w,f,f,f,b,f,f,f,w],
            [w,f,f,f,f,f,f,f,w],
            [w,f,f,f,o,f,f,f,w],
            [w,w,w,w,w,w,w,w,w],
        ],
        12: [
            [w,w,w,w,w,w,w,w,w],
            [w,o,f,f,f,f,f,o,w],
            [w,w,w,f,f,f,w,w,w],
            [w,f,b,f,f,f,b,f,w],
            [w,f,f,f,a,f,f,f,w],
            [w,f,f,f,b,f,f,f,w],
            [w,w,w,f,f,f,w,w,w],
            [w,f,f,f,o,f,f,f,w],
            [w,w,w,w,w,w,w,w,w],
        ],
        13: [
            [w,w,w,w,w,w,w,w,w],
            [w,o,f,f,f,f,f,o,w],
            [w,f,f,f,f,f,f,f,w],
            [w,f,b,f,w,f,b,f,w],
            [w,f,f,f,a,f,f,f,w],
            [w,f,f,f,b,f,f,f,w],
            [w,f,f,f,f,f,f,f,w],
            [w,o,f,f,f,f,f,f,w],
            [w,w,w,w,w,w,w,w,w],
        ],
        14: [
            [w,w,w,w,w,w,w,w,w],
            [w,o,f,f,f,f,f,o,w],
            [w,f,f,w,f,w,f,f,w],
            [w,f,f,b,f,b,f,f,w],
            [w,f,f,f,a,f,f,f,w],
            [w,f,f,f,b,f,f,f,w],
            [w,f,f,w,f,w,f,f,w],
            [w,f,f,f,o,f,f,f,w],
            [w,w,w,w,w,w,w,w,w],
        ],
        15: [
            [w,w,w,w,w,w,w,w,w],
            [w,o,f,f,f,f,f,o,w],
            [w,f,f,f,w,f,f,f,w],
            [w,f,b,f,f,f,b,f,w],
            [w,f,f,f,a,f,f,f,w],
            [w,f,f,f,b,f,f,f,w],
            [w,f,f,f,w,f,f,f,w],
            [w,f,f,f,o,f,f,f,w],
            [w,w,w,w,w,w,w,w,w],
        ],
        16: [
            [w,w,w,w,w,w,w,w,w],
            [w,o,f,f,f,f,f,o,w],
            [w,f,w,f,f,f,w,f,w],
            [w,f,b,f,f,f,b,f,w],
            [w,f,f,f,a,f,f,f,w],
            [w,f,f,f,b,f,f,f,w],
            [w,f,w,f,f,f,w,f,w],
            [w,f,f,f,o,f,f,f,w],
            [w,w,w,w,w,w,w,w,w],
        ],
        17: [
            [w,w,w,w,w,w,w,w,w],
            [w,o,f,f,f,f,f,f,w],
            [w,f,f,f,w,f,f,f,w],
            [w,f,b,f,f,f,b,f,w],
            [w,f,f,f,a,f,f,f,w],
            [w,f,f,f,b,f,f,f,w],
            [w,f,f,f,w,f,f,f,w],
            [w,f,f,f,o,f,f,o,w],
            [w,w,w,w,w,w,w,w,w],
        ],
        18: [
            [w,w,w,w,w,w,w,w,w],
            [w,o,f,f,f,f,f,o,w],
            [w,f,f,f,f,f,f,f,w],
            [w,w,f,b,f,b,f,w,w],
            [w,f,f,f,a,f,f,f,w],
            [w,w,f,f,b,f,f,w,w],
            [w,f,f,f,f,f,f,f,w],
            [w,f,f,f,o,f,f,f,w],
            [w,w,w,w,w,w,w,w,w],
        ],
        19: [
            [w,w,w,w,w,w,w,w,w],
            [w,o,f,f,f,f,f,o,w],
            [w,f,f,w,f,w,f,f,w],
            [w,f,b,f,f,f,b,f,w],
            [w,f,f,f,a,f,f,f,w],
            [w,f,f,f,b,f,f,f,w],
            [w,f,f,w,f,w,f,f,w],
            [w,o,f,f,f,f,f,f,w],
            [w,w,w,w,w,w,w,w,w],
        ],
        20: [
            [w,w,w,w,w,w,w,w,w],
            [w,o,f,f,f,f,f,o,w],
            [w,f,w,w,f,w,w,f,w],
            [w,f,f,b,f,b,f,f,w],
            [w,f,f,f,a,f,f,f,w],
            [w,f,f,b,f,b,f,f,w],
            [w,f,w,w,f,w,w,f,w],
            [w,o,f,f,f,f,f,o,w],
            [w,w,w,w,w,w,w,w,w],
        ],
    }


# ── Juego principal ───────────────────────────────────────────────────────────
class SokobanGame:
    """Clase central del juego. Contiene el bucle principal, todos los datos de
    la partida en curso y los métodos de dibujo de cada pantalla.

    ARQUITECTURA: el juego es una MÁQUINA DE ESTADOS. En cada momento está en
    uno de los estados S_* (menú, jugando, tienda, etc.). El bucle principal
    run() mira el estado actual, llama al método _draw_* correspondiente, y
    procesa los eventos con _handle_event(). Cambiar de pantalla = cambiar
    self.state.

    Los métodos _draw_* además de pintar REGISTRAN los botones del frame en
    self.buttons; el manejador de eventos recorre esa lista para detectar clics.
    """

    # ── Identificadores de los estados (pantallas) del juego ──────────────────
    S_MENU      = 'menu'        # menú principal
    S_NAME      = 'name'        # introducir/elegir nombre de jugador
    S_SELECT    = 'select'      # mapa de selección de nivel
    S_PLAY      = 'play'        # jugando un nivel
    S_WIN_LEVEL = 'win_level'   # pantalla de nivel completado
    S_WIN_GAME  = 'win_game'    # pantalla de juego completado (los 20 niveles)
    S_RANK      = 'rank'        # ranking global de jugadores
    S_SHOP      = 'shop'        # tienda (personajes, cajas, items)
    S_LEVEL_TOP = 'level_top'   # tabla de récords de un nivel concreto
    S_SETTINGS  = 'settings'    # ajustes de audio

    def __init__(self):
        """Inicializa pygame, carga recursos y deja el juego listo en el menú."""
        pg.init()
        pg.font.init()
        self.window = pg.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pg.display.set_caption('Sokoban')

        # Fuentes
        self.font_title = pg.font.SysFont("Arial", 56, bold=True)
        self.font_big = pg.font.SysFont("Arial", 36, bold=True)
        self.font_med = pg.font.SysFont("Arial", 26, bold=True)
        self.font_small = pg.font.SysFont("Arial", 20)
        self.font_tiny = pg.font.SysFont("Arial", 16)

        # Sistema de sonidos
        self.sounds = SoundSystem()

        # Datos
        self.data = GameData()
        self.levels = get_levels()
        self.total_levels = len(self.levels)

        # Aplicar settings persistentes al SoundSystem
        self.sounds.set_music_enabled(self.data.get_setting('music_enabled'))
        self.sounds.set_sfx_enabled(self.data.get_setting('sfx_enabled'))
        self.sounds.set_music_volume(self.data.get_setting('music_volume'))
        self.sounds.set_sfx_volume(self.data.get_setting('sfx_volume'))

        # Fondo del menú: intenta cargar bg_menu.png (mismo dir del script)
        self.bg_menu = None
        for candidate in ('bg_menu.png',
                          os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       'bg_menu.png')):
            try:
                img = pg.image.load(candidate).convert()
                if img.get_size() != (WINDOW_SIZE, WINDOW_SIZE):
                    img = pg.transform.smoothscale(img, (WINDOW_SIZE, WINDOW_SIZE))
                self.bg_menu = img
                break
            except (pg.error, FileNotFoundError):
                continue

        # Skin del cerdo desde archivo (sobrescribe el cerdo vectorial)
        self.pig_image = None
        for candidate in ('pig_skin.png',
                          os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       'pig_skin.png')):
            try:
                img = pg.image.load(candidate).convert_alpha()
                self.pig_image = img
                break
            except (pg.error, FileNotFoundError):
                continue

        # Tiles fijos
        self._build_static_tiles()

        # Skins cache
        self._skin_cache = {}

        # Estado
        self.state = self.S_MENU
        self.player_name = None
        self.name_input = ''
        self.name_error = ''

        # Datos durante juego
        self.current_level = 1
        self.main_level = [['' for _ in range(GRID)] for _ in range(GRID)]
        self.player_moves = 0
        self.player_pushes = 0
        self.undo_stack = []  # snapshots para deshacer
        self.last_rewards = []  # mensajes de recompensa
        self.solver_thread = None
        self.solver_result = None  # int o None
        self.is_replay = False  # si el nivel es repetición

        # Animación de movimiento (None o dict con campos
        # start_ms, duration_ms, p_from, p_to, b_from, b_to)
        self.animation = None
        # Timer del nivel
        self.level_start_ms = 0
        self.last_time_ms = 0  # tiempo del último intento (al ganar)
        self.last_position = None  # posición en leaderboard tras ganar

        # Scroll para listas
        self.rank_scroll = 0
        self.shop_scroll = 0
        self.select_scroll = 0
        self.level_top_target = None  # nivel a mostrar en S_LEVEL_TOP
        self.shop_tab = 'char'  # 'char' | 'box' | 'item'

        # Uso de items durante la partida
        # 'idle' / 'targeting_wall' (bomba) / 'targeting_box' (mágico paso 1)
        # / 'targeting_target' (mágico paso 2)
        self.item_mode = 'idle'
        self.magic_pending_box = None  # (x, y) de caja elegida
        self.items_used_run = False  # se usó algún item en este intento
        self.explosion = None  # animación de bomba activa

        # Confirmación de borrado de jugador
        self.delete_confirm = None  # nombre pendiente de confirmar borrado

        # Cache de mini-previsualizaciones para el selector
        self._mini_board_cache = {}

        # Botones (se reconstruyen cada frame)
        self.buttons = []  # lista de (rect, callback)

    # ── Tiles estáticos ─────────────────────────────────────────────────────
    def _build_static_tiles(self):
        """Pre-renderiza los tiles que nunca cambian (suelo, pared, objetivo,
        galaxia) una sola vez, para no redibujarlos cada frame."""
        sz = TILE
        # galaxy
        self.galaxy = _surf(sz)
        self.galaxy.fill((15, 15, 35))
        for _ in range(40):
            x, y = random.randint(0, sz-1), random.randint(0, sz-1)
            pg.draw.circle(self.galaxy, (200, 200, 255), (x, y), 1)
        # wall
        self.wall = _surf(sz)
        self.wall.fill((90, 95, 110))
        pg.draw.rect(self.wall, (50, 55, 65), (0, 0, sz, sz), 3)
        pg.draw.line(self.wall, (110, 115, 130), (sz//2, 0), (sz//2, sz), 1)
        pg.draw.line(self.wall, (110, 115, 130), (0, sz//2), (sz, sz//2), 1)
        # floor
        self.floor = _surf(sz)
        self.floor.fill((200, 195, 180))
        pg.draw.rect(self.floor, (185, 180, 165), (0, 0, sz, sz), 1)
        # target
        self.target = _surf(sz)
        self.target.fill((200, 195, 180))
        pg.draw.circle(self.target, (180, 80, 80), (sz//2, sz//2), sz//3, 3)
        pg.draw.circle(self.target, (180, 80, 80), (sz//2, sz//2), sz//8)

    # ── Helpers UI ──────────────────────────────────────────────────────────
    def _draw_text(self, text, font, color, pos, center=False):
        """Dibuja una cadena de texto. Si center=True, (x,y) es el centro;
        si no, la esquina superior izquierda."""
        surf = font.render(text, True, color)
        rect = surf.get_rect()
        if center:
            rect.center = pos
        else:
            rect.topleft = pos
        self.window.blit(surf, rect)
        return rect

    # ── Iconos vectoriales (independientes de la fuente) ────────────────────
    def _icon_coin(self, cx, cy, r):
        """Dibuja el icono vectorial de una moneda en (cx, cy)."""
        pg.draw.circle(self.window, (255, 200, 80), (cx, cy), r)
        pg.draw.circle(self.window, (180, 130, 30), (cx, cy), r, 2)
        pg.draw.line(self.window, (180, 130, 30),
                     (cx, cy - r//2), (cx, cy + r//2), 2)

    def _icon_trophy(self, cx, cy, r):
        """Dibuja el icono vectorial de un trofeo en (cx, cy)."""
        # copa
        pg.draw.rect(self.window, (255, 200, 80),
                     (cx - r, cy - r, 2*r, int(r*1.3)), border_radius=r//3)
        pg.draw.rect(self.window, (180, 130, 30),
                     (cx - r, cy - r, 2*r, int(r*1.3)), 2, border_radius=r//3)
        # base
        pg.draw.rect(self.window, (180, 130, 30),
                     (cx - r//2, cy + r//3, r, r//3))
        pg.draw.rect(self.window, (140, 100, 20),
                     (cx - r, cy + r*2//3, 2*r, r//3), border_radius=2)
        # asas
        pg.draw.arc(self.window, (180, 130, 30),
                    (cx - int(r*1.5), cy - r, r, r), 1.0, 2.5, 3)
        pg.draw.arc(self.window, (180, 130, 30),
                    (cx + r//2, cy - r, r, r), 0.6, 2.1, 3)

    def _icon_cart(self, cx, cy, r):
        """Dibuja el icono vectorial de un carrito de compra."""
        # cesta
        pg.draw.polygon(self.window, (90, 220, 130), [
            (cx - r, cy - r//2),
            (cx + r, cy - r//2),
            (cx + r*3//4, cy + r//3),
            (cx - r*3//4, cy + r//3)])
        pg.draw.polygon(self.window, (50, 140, 80), [
            (cx - r, cy - r//2),
            (cx + r, cy - r//2),
            (cx + r*3//4, cy + r//3),
            (cx - r*3//4, cy + r//3)], 2)
        # mango
        pg.draw.line(self.window, (50, 140, 80),
                     (cx - r, cy - r//2), (cx - r*3//2, cy - r), 2)
        # ruedas
        pg.draw.circle(self.window, (50, 140, 80), (cx - r//3, cy + r//2), max(2, r//4))
        pg.draw.circle(self.window, (50, 140, 80), (cx + r//3, cy + r//2), max(2, r//4))

    def _icon_lock(self, cx, cy, r):
        """Dibuja el icono vectorial de un candado (nivel bloqueado)."""
        # cuerpo
        body = pg.Rect(cx - r, cy - r//4, 2*r, r + r//4)
        pg.draw.rect(self.window, (200, 200, 215), body, border_radius=3)
        pg.draw.rect(self.window, (60, 60, 75), body, 2, border_radius=3)
        # arco
        pg.draw.arc(self.window, (200, 200, 215),
                    (cx - r*3//4, cy - r*3//2, int(r*1.5), int(r*1.5)),
                    0.2, 3.0, 4)
        pg.draw.circle(self.window, (60, 60, 75),
                       (cx, cy + r//4), max(2, r//5))

    def _icon_check(self, cx, cy, r, color=GOOD):
        """Dibuja una marca de verificación (nivel completado)."""
        pg.draw.line(self.window, color,
                     (cx - r, cy), (cx - r//3, cy + r*2//3), 4)
        pg.draw.line(self.window, color,
                     (cx - r//3, cy + r*2//3), (cx + r, cy - r*2//3), 4)

    def _icon_gear(self, cx, cy, r):
        """Dibuja el icono vectorial de un engranaje (ajustes)."""
        # ocho dientes alrededor de un círculo central
        for i in range(8):
            ang = i * math.pi / 4
            x1 = cx + math.cos(ang) * r * 0.7
            y1 = cy + math.sin(ang) * r * 0.7
            x2 = cx + math.cos(ang) * (r + 2)
            y2 = cy + math.sin(ang) * (r + 2)
            pg.draw.line(self.window, (180, 180, 200),
                         (x1, y1), (x2, y2), 4)
        pg.draw.circle(self.window, (180, 180, 200), (cx, cy), int(r * 0.7))
        pg.draw.circle(self.window, BG_DARK, (cx, cy), int(r * 0.3))

    def _icon_speaker(self, cx, cy, r, *, muted=False):
        """Dibuja el icono de un altavoz; con muted=True le añade una X roja."""
        # cono del altavoz
        pg.draw.polygon(self.window, (200, 200, 215), [
            (cx - r, cy - r//2), (cx - r//4, cy - r//2),
            (cx + r//4, cy - r), (cx + r//4, cy + r),
            (cx - r//4, cy + r//2), (cx - r, cy + r//2)])
        if muted:
            pg.draw.line(self.window, BAD,
                         (cx - r, cy - r), (cx + r, cy + r), 3)
        else:
            # dos ondas
            pg.draw.arc(self.window, (200, 200, 215),
                        (cx + r//2 - 4, cy - r, r, r*2), -1.0, 1.0, 2)
            pg.draw.arc(self.window, (200, 200, 215),
                        (cx + r - 4, cy - r*3//2, int(r*1.5), r*3),
                        -1.0, 1.0, 2)

    def _icon_trash(self, cx, cy, r):
        """Dibuja el icono vectorial de una papelera (borrar)."""
        # tapa
        pg.draw.rect(self.window, (200, 90, 90),
                     (cx - r, cy - r, 2*r, r//3), border_radius=2)
        # asa
        pg.draw.rect(self.window, (200, 90, 90),
                     (cx - r//2, cy - r*5//4, r, r//4), border_radius=2)
        # cuerpo
        body = pg.Rect(cx - r*4//5, cy - r*2//3, r*8//5, int(r*1.5))
        pg.draw.rect(self.window, (200, 90, 90), body, border_radius=2)
        pg.draw.rect(self.window, (120, 30, 30), body, 2, border_radius=2)
        # líneas verticales
        for dx in (-r//3, 0, r//3):
            pg.draw.line(self.window, (120, 30, 30),
                         (cx + dx, cy - r//3),
                         (cx + dx, cy + r*2//3), 2)

    def _draw_coin_label(self, value, pos, font=None, center=False, color=ACCENT):
        """Dibuja '🪙 N' usando un icono vectorial."""
        font = font or self.font_med
        text_surf = font.render(str(value), True, color)
        icon_r = font.get_height() // 3
        gap = 8
        total_w = icon_r*2 + gap + text_surf.get_width()
        if center:
            x = pos[0] - total_w // 2
            y_text = pos[1] - text_surf.get_height() // 2
        else:
            x, y_text = pos
        cy = y_text + text_surf.get_height() // 2
        self._icon_coin(x + icon_r, cy, icon_r)
        self.window.blit(text_surf, (x + icon_r*2 + gap, y_text))

    def _get_skin(self, kind, key, on_target=False):
        """Devuelve la superficie de un skin (personaje o caja), con cache.
        Para el cerdo usa la imagen pig_skin.png si está disponible."""
        cache_key = (kind, key, on_target)
        if cache_key in self._skin_cache:
            return self._skin_cache[cache_key]
        # Caso especial: cerdo desde imagen externa
        if kind == 'char' and key == 'pig' and self.pig_image is not None:
            if self.pig_image.get_size() == (TILE, TILE):
                surf = self.pig_image.copy()
            else:
                surf = pg.transform.smoothscale(self.pig_image, (TILE, TILE))
            if on_target:
                surf = add_target_overlay(surf, TILE)
            self._skin_cache[cache_key] = surf
            return surf
        registry = CHARACTER_SKINS if kind == 'char' else BOX_SKINS
        info = registry.get(key) or registry[list(registry)[0]]
        surf = info['maker'](TILE)
        if on_target:
            surf = add_target_overlay(surf, TILE)
        self._skin_cache[cache_key] = surf
        return surf

    def _get_mini_board(self, level_num, tile_size=20):
        """Genera una miniatura del tablero de un nivel (para la vista previa
        del selector de niveles). Resultado cacheado."""
        key = (level_num, tile_size)
        if key in self._mini_board_cache:
            return self._mini_board_cache[key]
        grid = self.levels[level_num]
        surf = pg.Surface((GRID * tile_size, GRID * tile_size))
        surf.fill((20, 20, 30))
        colors = {
            'g': (15, 15, 35),
            'w': (90, 95, 110),
            'f': (200, 195, 180),
            'o': (200, 195, 180),
            'b': (200, 195, 180),
            'a': (200, 195, 180),
        }
        for y in range(GRID):
            for x in range(GRID):
                cell = grid[y][x]
                pg.draw.rect(surf, colors.get(cell, (100, 100, 100)),
                             (x*tile_size, y*tile_size, tile_size, tile_size))
                if cell == 'o':
                    pg.draw.circle(surf, (180, 80, 80),
                                   (x*tile_size + tile_size//2,
                                    y*tile_size + tile_size//2),
                                   tile_size//3, 2)
                elif cell == 'b':
                    pg.draw.rect(surf, (190, 140, 60),
                                 (x*tile_size + 2, y*tile_size + 2,
                                  tile_size - 4, tile_size - 4))
                elif cell == 'a':
                    pg.draw.circle(surf, (240, 160, 160),
                                   (x*tile_size + tile_size//2,
                                    y*tile_size + tile_size//2),
                                   tile_size//3)
        pg.draw.rect(surf, (90, 100, 130), surf.get_rect(), 2)
        self._mini_board_cache[key] = surf
        return surf

    def _button(self, rect, label, callback, *,
                bg=BG_PANEL, fg=TEXT, border=PRIMARY, hover_bg=None,
                font=None, disabled=False):
        """Dibuja un botón rectangular y REGISTRA su (rect, callback) en
        self.buttons para que _handle_event detecte el clic."""
        font = font or self.font_med
        mouse = pg.mouse.get_pos()
        hovered = rect.collidepoint(mouse) and not disabled
        if disabled:
            actual_bg = (40, 40, 50)
            actual_fg = TEXT_DIM
            actual_border = (60, 60, 70)
        elif hovered:
            actual_bg = hover_bg or border
            actual_fg = BG_DARK
            actual_border = border
        else:
            actual_bg = bg
            actual_fg = fg
            actual_border = border

        pg.draw.rect(self.window, actual_bg, rect, border_radius=10)
        pg.draw.rect(self.window, actual_border, rect, 2, border_radius=10)
        surf = font.render(label, True, actual_fg)
        self.window.blit(surf, surf.get_rect(center=rect.center))

        if not disabled:
            self.buttons.append((rect, callback))

    def _clear(self):
        """Rellena toda la ventana con el color de fondo."""
        self.window.fill(BG_DARK)
        self.buttons = []

    # ── Solver asíncrono ────────────────────────────────────────────────────
    def _start_solver(self, level_num):
        """Lanza el solver en un HILO aparte para no congelar el juego.
        El resultado (óptimo) se recoge luego en self.solver_result."""
        cached = self.data.get_min_moves(level_num)
        if cached is not None:
            self.solver_result = cached
            self.solver_thread = None
            return
        self.solver_result = None
        level = self.levels[level_num]

        def run():
            res = solve_level(level)
            self.solver_result = res
            if res is not None:
                self.data.set_min_moves(level_num, res)

        self.solver_thread = threading.Thread(target=run, daemon=True)
        self.solver_thread.start()

    # ── Cargar nivel ────────────────────────────────────────────────────────
    def _load_level(self, level_num):
        """Carga un nivel: copia su matriz, reinicia contadores, cronómetro,
        pila de deshacer y estado de items, y arranca el solver."""
        self.current_level = level_num
        grid = self.levels[level_num]
        for y in range(GRID):
            for x in range(GRID):
                self.main_level[y][x] = grid[y][x]
        self.player_moves = 0
        self.player_pushes = 0
        self.undo_stack = []
        self.animation = None
        self.level_start_ms = pg.time.get_ticks()
        self.last_time_ms = 0
        self.last_position = None
        self.item_mode = 'idle'
        self.magic_pending_box = None
        self.items_used_run = False
        self.explosion = None
        player = self.data.get_player(self.player_name)
        self.is_replay = level_num <= player.get('max_completed', 0)
        self._start_solver(level_num)
        self.last_rewards = []

    # ── Dibujo del tablero ──────────────────────────────────────────────────
    def _draw_board(self):
        """Dibuja el tablero completo celda a celda. Si hay una animación de
        movimiento activa, interpola la posición de jugador y caja."""
        player = self.data.get_player(self.player_name)
        char_key = player.get('character', 'pig')
        box_key = player.get('box_skin', 'wooden')

        char_surf = self._get_skin('char', char_key, False)
        char_ot_surf = self._get_skin('char', char_key, True)
        box_surf = self._get_skin('box', box_key, False)
        box_ot_surf = self._get_skin('box', box_key, True)

        # Si hay animación activa, calculamos t y las celdas que hay que
        # "limpiar" (no dibujar al jugador/caja en su posición final del grid)
        anim = self.animation
        anim_active = self._animation_active()
        skip_cells = set()  # celdas donde NO debemos pintar el contenido normal
        if anim_active:
            elapsed = pg.time.get_ticks() - anim['start_ms']
            t = elapsed / anim['duration_ms']
            t = max(0.0, min(1.0, t))
            # ease-out cubic
            t_eased = 1 - (1 - t) ** 3
            skip_cells.add(anim['p_from'])
            skip_cells.add(anim['p_to'])
            if anim['b_from']:
                skip_cells.add(anim['b_from'])
                skip_cells.add(anim['b_to'])
        else:
            t_eased = 1.0

        for y in range(GRID):
            for x in range(GRID):
                cell = self.main_level[y][x]
                pos = (x * TILE, y * TILE)
                if (x, y) in skip_cells:
                    # Pintar sólo el suelo/objetivo subyacente,
                    # los actores se pintan luego con interpolación
                    if cell in ('a',):
                        self.window.blit(self.floor, pos)
                    elif cell in ('aot',):
                        self.window.blit(self.target, pos)
                    elif cell == 'b':
                        self.window.blit(self.floor, pos)
                    elif cell == 'bot':
                        self.window.blit(self.target, pos)
                    elif cell == 'g':
                        self.window.blit(self.galaxy, pos)
                    elif cell == 'w':
                        self.window.blit(self.wall, pos)
                    elif cell == 'f':
                        self.window.blit(self.floor, pos)
                    elif cell == 'o':
                        self.window.blit(self.target, pos)
                    continue
                if cell == 'g':
                    self.window.blit(self.galaxy, pos)
                elif cell == 'w':
                    self.window.blit(self.wall, pos)
                elif cell == 'f':
                    self.window.blit(self.floor, pos)
                elif cell == 'o':
                    self.window.blit(self.target, pos)
                elif cell == 'b':
                    self.window.blit(self.floor, pos)
                    self.window.blit(box_surf, pos)
                elif cell == 'bot':
                    self.window.blit(self.target, pos)
                    self.window.blit(box_ot_surf, pos)
                elif cell == 'a':
                    self.window.blit(self.floor, pos)
                    self.window.blit(char_surf, pos)
                elif cell == 'aot':
                    self.window.blit(self.target, pos)
                    self.window.blit(char_ot_surf, pos)

        # Dibujar actores animados encima
        if anim_active:
            # caja primero (jugador delante)
            if anim['b_from']:
                bx = anim['b_from'][0] + (anim['b_to'][0] - anim['b_from'][0]) * t_eased
                by = anim['b_from'][1] + (anim['b_to'][1] - anim['b_from'][1]) * t_eased
                bdx, bdy = anim['b_to']
                target_cell = self.main_level[bdy][bdx]
                surf = box_ot_surf if target_cell == 'bot' else box_surf
                self.window.blit(surf, (bx * TILE, by * TILE))
            # jugador
            px = anim['p_from'][0] + (anim['p_to'][0] - anim['p_from'][0]) * t_eased
            py = anim['p_from'][1] + (anim['p_to'][1] - anim['p_from'][1]) * t_eased
            pdx, pdy = anim['p_to']
            cell = self.main_level[pdy][pdx]
            surf = char_ot_surf if cell == 'aot' else char_surf
            self.window.blit(surf, (px * TILE, py * TILE))

    # ── Movimientos ─────────────────────────────────────────────────────────
    def _player_pos(self):
        """Devuelve la posición (x, y) actual del jugador en el grid."""
        for y in range(GRID):
            for x in range(GRID):
                if self.main_level[y][x] in ('a', 'aot'):
                    return (x, y)
        return (0, 0)

    def _move(self, dx, dy):
        """Intenta mover al jugador (dx, dy). Gestiona el empuje de cajas, la
        pila de deshacer, la animación y los sonidos. Devuelve True si
        el movimiento fue válido."""
        px, py = self._player_pos()
        nx, ny = px + dx, py + dy
        sx, sy = px + 2*dx, py + 2*dy
        if not (0 <= nx < GRID and 0 <= ny < GRID):
            return False
        cell = self.main_level[ny][nx]
        if cell in ('w', 'g'):
            return False
        pushed = False
        if cell in ('b', 'bot'):
            if not (0 <= sx < GRID and 0 <= sy < GRID):
                return False
            ahead = self.main_level[sy][sx]
            if ahead not in ('f', 'o'):
                return False
            pushed = True

        # snapshot ANTES de modificar (para undo)
        snapshot = [row[:] for row in self.main_level]
        self.undo_stack.append((snapshot, self.player_moves, self.player_pushes))
        # límite por si el jugador se entretiene mucho
        if len(self.undo_stack) > 500:
            self.undo_stack = self.undo_stack[-500:]

        # Iniciar animación visual antes de mutar el grid lógico
        self.animation = {
            'start_ms': pg.time.get_ticks(),
            'duration_ms': MOVE_ANIM_MS,
            'p_from': (px, py),
            'p_to': (nx, ny),
            'b_from': (nx, ny) if pushed else None,
            'b_to': (sx, sy) if pushed else None,
        }

        if pushed:
            ahead = self.main_level[sy][sx]
            self.main_level[sy][sx] = 'bot' if ahead == 'o' else 'b'
            self.player_pushes += 1
            self.sounds.play('push')
        else:
            self.sounds.play('step')

        # vaciar celda actual
        cur = self.main_level[py][px]
        self.main_level[py][px] = 'o' if cur == 'aot' else 'f'
        # ocupar nueva celda
        self.main_level[ny][nx] = 'aot' if cell in ('o', 'bot') else 'a'
        self.player_moves += 1
        return True

    def _animation_active(self):
        """True si hay una animación de movimiento en curso. Si ya terminó,
        la limpia y devuelve False."""
        if self.animation is None:
            return False
        elapsed = pg.time.get_ticks() - self.animation['start_ms']
        if elapsed >= self.animation['duration_ms']:
            self.animation = None
            return False
        return True

    # ── Items en partida ────────────────────────────────────────────────────
    def _click_to_cell(self, mouse_pos):
        """Convierte coordenadas de ratón a celda (x, y) del tablero.
        Devuelve None si no está sobre el tablero (queda bajo el HUD)."""
        mx, my = mouse_pos
        # El HUD ocupa los primeros 54 px verticales
        if my < 54:
            return None
        cx, cy = mx // TILE, my // TILE
        if 0 <= cx < GRID and 0 <= cy < GRID:
            return (cx, cy)
        return None

    def _try_use_item_at(self, cell):
        """Ejecuta el item según el modo actual y la celda clicada."""
        if cell is None or self.item_mode == 'idle':
            return
        x, y = cell
        target_cell = self.main_level[y][x]

        if self.item_mode == 'targeting_wall':
            if target_cell != 'w':
                self.sounds.play('error')
                return
            # No volar paredes del borde (rompería el recinto)
            if x == 0 or y == 0 or x == GRID - 1 or y == GRID - 1:
                self.sounds.play('error')
                return
            if not self.data.use_item(self.player_name, 'bomb'):
                self.sounds.play('error')
                return
            # convertir en suelo
            self.main_level[y][x] = 'f'
            self.items_used_run = True
            self.item_mode = 'idle'
            # Animación de explosión y sonido
            self.explosion = {
                'cx': x * TILE + TILE // 2,
                'cy': y * TILE + TILE // 2,
                'start_ms': pg.time.get_ticks(),
                'duration_ms': 600,
            }
            self.sounds.play('explosion')
            return

        if self.item_mode == 'targeting_box':
            if target_cell not in ('b', 'bot'):
                self.sounds.play('error')
                return
            self.magic_pending_box = (x, y)
            self.item_mode = 'targeting_target'
            self.sounds.play('click')
            return

        if self.item_mode == 'targeting_target':
            # admitir cualquier target libre ('o' o 'aot' no, 'bot' es ocupado)
            if target_cell != 'o':
                self.sounds.play('error')
                return
            if self.magic_pending_box is None:
                self.item_mode = 'idle'
                return
            if not self.data.use_item(self.player_name, 'magic'):
                self.sounds.play('error')
                return
            # quitar caja origen
            bx, by = self.magic_pending_box
            src = self.main_level[by][bx]
            self.main_level[by][bx] = 'o' if src == 'bot' else 'f'
            # colocar en destino
            self.main_level[y][x] = 'bot'
            self.items_used_run = True
            self.item_mode = 'idle'
            self.magic_pending_box = None
            self.sounds.play('win_level')
            # comprobar victoria inmediata
            if self._check_win():
                self.last_time_ms = pg.time.get_ticks() - self.level_start_ms
                if self.solver_thread is not None:
                    self.solver_thread.join(timeout=1.0)
                prev_max = self.data.get_player(self.player_name).get('max_completed', 0)
                self._award_rewards()
                if self.current_level == self.total_levels and prev_max < self.total_levels:
                    self.sounds.play('win_game')
                    self.state = self.S_WIN_GAME
                else:
                    self.sounds.play('win_level')
                    self.state = self.S_WIN_LEVEL
            return

    def _undo(self):
        """Deshace el último movimiento restaurando el snapshot guardado en
        la pila de deshacer."""
        if not self.undo_stack:
            return False
        snapshot, moves, pushes = self.undo_stack.pop()
        self.main_level = snapshot
        self.player_moves = moves
        self.player_pushes = pushes
        self.animation = None  # cancelar cualquier animación pendiente
        return True

    def _check_win(self):
        """True si todas las cajas están sobre objetivos (nivel resuelto)."""
        for y in range(GRID):
            for x in range(GRID):
                if self.main_level[y][x] == 'b':
                    return False
        return True

    # ── Recompensas al ganar ────────────────────────────────────────────────
    def _award_rewards(self):
        """Calcula y otorga las monedas ganadas al completar un nivel, las
        registra en la tabla de récords y prepara los mensajes de
        recompensa para la pantalla de victoria."""
        player = self.data.get_player(self.player_name)
        rewards = []
        lvl = self.current_level
        time_ms = self.last_time_ms  # se rellena justo antes de llamar

        first_time = lvl > player.get('max_completed', 0)

        # Comparar con récord personal previo (para bonus de replay)
        prev_record = player.get('records', {}).get(str(lvl))

        if first_time:
            base = 5
            rewards.append((f'+{base} monedas por completar el nivel', base))

            if lvl % 10 == 0:
                rewards.append(('+20 monedas (bono de 10 niveles)', 20))

            optimal_levels = list(player.get('optimal_levels', []))
            if self.solver_result is not None \
               and self.player_moves == self.solver_result \
               and not self.items_used_run:
                if lvl not in optimal_levels:
                    rewards.append(('+40 monedas por ruta óptima', 40))
                    optimal_levels.append(lvl)

            if lvl == self.total_levels:
                rewards.append(('+100 monedas ¡todos los niveles!', 100))

            if self.items_used_run:
                rewards.append(('(usaste items: sin bonus de óptimo)', 0))

            total_coins = sum(c for _, c in rewards)
            new_coins = player.get('coins', 0) + total_coins
            new_max = lvl
            new_level = lvl + 1

            self.data.update_player(self.player_name,
                                    coins=new_coins,
                                    max_completed=new_max,
                                    level=new_level,
                                    optimal_levels=optimal_levels)
        else:
            # Replay
            optimal_levels = list(player.get('optimal_levels', []))
            if self.solver_result is not None \
               and self.player_moves == self.solver_result \
               and not self.items_used_run:
                if lvl not in optimal_levels:
                    rewards.append(('+40 monedas por ruta óptima (primera vez)', 40))
                    optimal_levels.append(lvl)
                    new_coins = player.get('coins', 0) + 40
                    self.data.update_player(self.player_name,
                                            coins=new_coins,
                                            optimal_levels=optimal_levels)

            # bonus por mejorar récord personal (sólo en replay y sin items)
            extra = 0
            if prev_record is not None and not self.items_used_run:
                if self.player_moves < prev_record['moves']:
                    rewards.append(('+5 monedas: récord personal de movimientos',
                                    5))
                    extra += 5
                if time_ms < prev_record.get('time_ms', 10**9):
                    rewards.append(('+5 monedas: récord personal de tiempo', 5))
                    extra += 5
            if extra:
                self.data.update_player(
                    self.player_name,
                    coins=self.data.get_player(self.player_name).get('coins', 0)
                          + extra)
            if self.items_used_run:
                rewards.append(('(usaste items: sin bonus)', 0))
            if not rewards:
                rewards.append(('Nivel repetido (sin monedas)', 0))

        # Registrar el run en el leaderboard del nivel
        position, _, _ = self.data.submit_level_run(
            self.player_name, lvl,
            self.player_moves, self.player_pushes, time_ms)
        self.last_position = position
        self.last_rewards = rewards

    # ── Estado: MENÚ ────────────────────────────────────────────────────────
    def _draw_menu(self):
        """Dibuja el menú principal (estado S_MENU)."""
        self._clear()
        # arrancar ambiente de granja si no está sonando
        self.sounds.play_loop('farm_ambient')

        if self.bg_menu is not None:
            # Fondo desde imagen + overlay oscuro vertical para legibilidad
            self.window.blit(self.bg_menu, (0, 0))
            overlay = pg.Surface((WINDOW_SIZE, WINDOW_SIZE), pg.SRCALPHA)
            # Banda oscura más fuerte donde van los botones (de y=210 a y=600)
            for y in range(WINDOW_SIZE):
                if 200 <= y <= 610:
                    # más oscuro en el centro de la banda
                    t = 1.0 - abs(y - 405) / 205
                    alpha = int(110 + 80 * t)
                else:
                    alpha = 60
                pg.draw.line(overlay, (0, 0, 0, alpha), (0, y), (WINDOW_SIZE, y))
            self.window.blit(overlay, (0, 0))
        else:
            # Fallback: campo de estrellas (preservando estado de random)
            _bk = random.getstate()
            random.seed(42)
            for _ in range(80):
                x = random.randint(0, WINDOW_SIZE)
                y = random.randint(0, WINDOW_SIZE)
                pg.draw.circle(self.window, (40, 40, 60), (x, y), 1)
            random.setstate(_bk)

        # Título
        self._draw_text('SOKOBAN', self.font_title, ACCENT,
                        (WINDOW_SIZE//2, 130), center=True)
        self._draw_text('20 niveles · monedas · tienda',
                        self.font_small, TEXT_DIM,
                        (WINDOW_SIZE//2, 180), center=True)

        # Botones con iconos vectoriales al lado izquierdo
        # Cinco botones compactos (alto 58 px, separación 70 px)
        bw, bh = 320, 58
        bx = WINDOW_SIZE//2 - bw//2
        y0 = 240
        gap = 70

        def play_btn():
            r = pg.Rect(bx, y0, bw, bh)
            self._button(r, '   JUGAR', lambda: self._enter_name_state())
            pg.draw.polygon(self.window, TEXT,
                            [(r.x + 30, r.y + 16), (r.x + 30, r.y + 42),
                             (r.x + 55, r.y + 29)])

        def rank_btn():
            r = pg.Rect(bx, y0 + gap, bw, bh)
            self._button(r, '   RANKING',
                         lambda: self._goto(self.S_RANK), border=ACCENT)
            self._icon_trophy(r.x + 40, r.y + 29, 14)

        def shop_btn():
            r = pg.Rect(bx, y0 + 2*gap, bw, bh)
            self._button(r, '   TIENDA',
                         lambda: self._goto_shop(), border=GOOD)
            self._icon_cart(r.x + 40, r.y + 29, 14)

        def settings_btn():
            r = pg.Rect(bx, y0 + 3*gap, bw, bh)
            self._button(r, '   AJUSTES',
                         lambda: self._goto(self.S_SETTINGS),
                         border=(180, 180, 200))
            self._icon_gear(r.x + 40, r.y + 29, 14)

        def exit_btn():
            r = pg.Rect(bx, y0 + 4*gap, bw, bh)
            self._button(r, 'SALIR', lambda: self._quit(), border=BAD)

        play_btn()
        rank_btn()
        shop_btn()
        settings_btn()
        exit_btn()

        self._draw_text('Pontevedra · Sokoban v2',
                        self.font_tiny, TEXT_DIM,
                        (WINDOW_SIZE//2, WINDOW_SIZE - 22), center=True)

    def _enter_name_state(self):
        """Cambia a la pantalla de introducir nombre."""
        self.name_input = ''
        self.name_error = ''
        if self.state == self.S_MENU:
            self.sounds.stop_loop()
        self.state = self.S_NAME

    def _goto(self, state):
        """Cambia el estado del juego. Si se sale del menú, corta la música."""
        # Si salimos del menú, parar el sonido ambiente
        if self.state == self.S_MENU and state != self.S_MENU:
            self.sounds.stop_loop()
        self.state = state
        if state == self.S_RANK:
            self.rank_scroll = 0

    def _goto_shop(self):
        """Abre la tienda (pide jugador si aún no hay)."""
        if self.player_name is None:
            self.name_error = 'Primero crea un jugador (JUGAR)'
            self.state = self.S_NAME
            self.name_input = ''
            self.sounds.stop_loop()
            return
        if self.state == self.S_MENU:
            self.sounds.stop_loop()
        self.shop_scroll = 0
        self.state = self.S_SHOP

    def _quit(self):
        """Cierra el juego limpiamente."""
        pg.quit()
        sys.exit()

    def _logout(self):
        """Cerrar sesión del jugador actual y volver al menú principal."""
        self.player_name = None
        self.state = self.S_MENU
        self.animation = None

    # ── Estado: NOMBRE ──────────────────────────────────────────────────────
    def _draw_name(self):
        """Dibuja la pantalla de creación/elección de jugador."""
        self._clear()
        self._draw_text('NUEVO JUGADOR', self.font_big, TEXT,
                        (WINDOW_SIZE//2, 130), center=True)
        self._draw_text('Escribe tu nombre y pulsa ENTER',
                        self.font_small, TEXT_DIM,
                        (WINDOW_SIZE//2, 175), center=True)

        # Input box
        box = pg.Rect(WINDOW_SIZE//2 - 220, 230, 440, 70)
        pg.draw.rect(self.window, BG_PANEL, box, border_radius=10)
        pg.draw.rect(self.window, PRIMARY, box, 2, border_radius=10)
        # texto + cursor
        shown = self.name_input + ('|' if (pg.time.get_ticks() // 500) % 2 == 0 else ' ')
        self._draw_text(shown, self.font_big, TEXT,
                        box.center, center=True)

        # Reglas
        self._draw_text('1-12 caracteres, sin duplicados',
                        self.font_small, TEXT_DIM,
                        (WINDOW_SIZE//2, 320), center=True)

        # Error
        if self.name_error:
            self._draw_text(self.name_error, self.font_small, BAD,
                            (WINDOW_SIZE//2, 360), center=True)

        # Jugadores existentes
        self._draw_text('Jugadores existentes:', self.font_small, TEXT_DIM,
                        (WINDOW_SIZE//2, 410), center=True)
        names = list(self.data.data['players'].keys())
        if not names:
            self._draw_text('(ninguno)', self.font_small, TEXT_DIM,
                            (WINDOW_SIZE//2, 440), center=True)
        else:
            shown_names = ', '.join(names[:8])
            if len(names) > 8:
                shown_names += f'... (+{len(names)-8})'
            self._draw_text(shown_names, self.font_small, TEXT,
                            (WINDOW_SIZE//2, 440), center=True)

        # Si el nombre ya existe, ofrecer iniciar sesión
        if self.name_input and self.data.player_exists(self.name_input):
            self._draw_text(f'"{self.name_input}" ya existe.',
                            self.font_small, ACCENT,
                            (WINDOW_SIZE//2, 490), center=True)
            self._button(pg.Rect(WINDOW_SIZE//2 - 150, 520, 300, 50),
                         'Continuar con este jugador',
                         lambda: self._login_existing(),
                         border=ACCENT, font=self.font_small)

        # Volver
        self._button(pg.Rect(20, WINDOW_SIZE - 70, 140, 50),
                     '< Menú', lambda: self._goto(self.S_MENU),
                     font=self.font_small)

    def _login_existing(self):
        """Inicia sesión con un jugador ya existente."""
        canonical = self.data.find_canonical(self.name_input)
        if canonical is None:
            self.name_error = 'Ese jugador ya no existe'
            return
        self.player_name = canonical
        self.name_input = ''
        self.name_error = ''
        self.state = self.S_SELECT
        self.select_scroll = 0

    def _submit_name(self):
        """Valida el nombre escrito y crea el jugador (o muestra el error)."""
        name = self.name_input.strip()
        if len(name) < 1:
            self.name_error = 'El nombre no puede estar vacío'
            self.sounds.play('error')
            return
        if len(name) > 12:
            self.name_error = 'Máximo 12 caracteres'
            self.sounds.play('error')
            return
        if self.data.player_exists(name):
            self.name_error = f'"{name}" ya existe (sin distinguir may/min)'
            self.sounds.play('error')
            return
        self.data.add_player(name)
        self.player_name = name
        self.name_input = ''
        self.name_error = ''
        self.state = self.S_SELECT
        self.select_scroll = 0

    # ── Estado: SELECCIÓN DE NIVELES (mapa lineal) ──────────────────────────
    def _draw_select(self):
        """Dibuja el mapa de selección de niveles (estado S_SELECT)."""
        self._clear()
        player = self.data.get_player(self.player_name)
        current = player['level']
        max_done = player.get('max_completed', 0)
        coins = player.get('coins', 0)
        all_done = max_done >= self.total_levels

        # Header
        self._draw_text(f'Jugador: {self.player_name}',
                        self.font_med, TEXT, (20, 18))
        # icono moneda alineado a la derecha (después del botón Menú)
        self._draw_coin_label(coins,
                              (WINDOW_SIZE - 230, 25),
                              font=self.font_med)
        # Cerrar sesión (desloguea y vuelve al menú)
        self._button(pg.Rect(WINDOW_SIZE - 150, 12, 130, 40),
                     'Cerrar sesión', lambda: self._logout(),
                     font=self.font_small, border=BAD)

        if all_done:
            self._draw_text(f'¡COMPLETADO! {self.total_levels}/{self.total_levels}',
                            self.font_small, GOOD, (20, 55))
        else:
            self._draw_text(f'Nivel actual: {current}/{self.total_levels}   '
                            f'Completados: {max_done}',
                            self.font_small, TEXT_DIM, (20, 55))

        # Mapa lineal: 4 columnas x 5 filas, conectados con líneas
        cols, rows = 4, 5
        node_size = 80
        top_pad = 110
        left_pad = 60
        gap_x = (WINDOW_SIZE - 2*left_pad - cols*node_size) // (cols - 1)
        gap_y = 30

        # primero dibujar conexiones
        for i in range(self.total_levels):
            r, c = i // cols, i % cols
            cx = left_pad + c*(node_size + gap_x) + node_size//2
            cy = top_pad + r*(node_size + gap_y) + node_size//2
            if i + 1 < self.total_levels:
                r2, c2 = (i+1)//cols, (i+1)%cols
                cx2 = left_pad + c2*(node_size + gap_x) + node_size//2
                cy2 = top_pad + r2*(node_size + gap_y) + node_size//2
                done = (i+1) <= max_done
                col = GOOD if done else (60, 60, 75)
                if r == r2:
                    pg.draw.line(self.window, col, (cx + node_size//2, cy),
                                 (cx2 - node_size//2, cy2), 4)
                else:
                    end_right = left_pad + (cols-1)*(node_size + gap_x) + node_size + 10
                    pg.draw.line(self.window, col,
                                 (cx + node_size//2, cy),
                                 (end_right, cy), 4)
                    pg.draw.line(self.window, col,
                                 (end_right, cy),
                                 (end_right, cy2), 4)
                    pg.draw.line(self.window, col,
                                 (end_right, cy2),
                                 (cx2 + node_size//2, cy2), 4)

        # ahora los nodos (botones)
        mouse_pos = pg.mouse.get_pos()
        hover_lvl = None
        for i in range(self.total_levels):
            lvl = i + 1
            r, c = i // cols, i % cols
            x = left_pad + c*(node_size + gap_x)
            y = top_pad + r*(node_size + gap_y)
            rect = pg.Rect(x, y, node_size, node_size)

            unlocked = lvl <= current
            done = lvl <= max_done

            if done:
                color = GOOD
                border = (60, 180, 100)
                label_color = BG_DARK
            elif unlocked:
                color = PRIMARY
                border = (60, 130, 220)
                label_color = TEXT
            else:
                color = LOCKED
                border = (50, 50, 65)
                label_color = TEXT_DIM

            hovered = rect.collidepoint(mouse_pos) and unlocked
            if hovered:
                pg.draw.rect(self.window, ACCENT, rect.inflate(6, 6),
                             border_radius=14)
                hover_lvl = lvl

            pg.draw.rect(self.window, color, rect, border_radius=12)
            pg.draw.rect(self.window, border, rect, 3, border_radius=12)

            if unlocked:
                self._draw_text(str(lvl), self.font_big, label_color,
                                rect.center, center=True)
                if done:
                    # check vectorial pequeño en esquina
                    self._icon_check(rect.right - 16, rect.top + 14, 6,
                                     color=BG_DARK)
                self.buttons.append((rect, (lambda l=lvl: self._play_level(l))))
            else:
                self._icon_lock(rect.centerx, rect.centery, 14)

        # Mini-preview al hover
        if hover_lvl is not None:
            preview = self._get_mini_board(hover_lvl, tile_size=18)
            pw, ph = preview.get_size()
            px = mouse_pos[0] + 18
            py = mouse_pos[1] + 18
            # ajustar para no salir de la pantalla
            if px + pw > WINDOW_SIZE - 6:
                px = mouse_pos[0] - pw - 18
            if py + ph > WINDOW_SIZE - 6:
                py = WINDOW_SIZE - ph - 6
            # fondo
            bg_rect = pg.Rect(px - 6, py - 24, pw + 12, ph + 30)
            pg.draw.rect(self.window, BG_PANEL, bg_rect, border_radius=8)
            pg.draw.rect(self.window, PRIMARY, bg_rect, 2, border_radius=8)
            self._draw_text(f'Nivel {hover_lvl}', self.font_small, TEXT,
                            (px, py - 22))
            self.window.blit(preview, (px, py))

    def _play_level(self, lvl):
        """Carga el nivel indicado y pasa al estado de juego."""
        if lvl < 1 or lvl > self.total_levels:
            return
        self._load_level(lvl)
        self.state = self.S_PLAY

    # ── Estado: JUGANDO ─────────────────────────────────────────────────────
    def _draw_play(self):
        """Dibuja la pantalla de juego: tablero, HUD y overlays."""
        self._clear()
        self._draw_board()

        # Overlay para modo de selección de item: resalta celdas válidas
        player = self.data.get_player(self.player_name)
        if self.item_mode != 'idle':
            valid_cells = []
            for y in range(GRID):
                for x in range(GRID):
                    cell = self.main_level[y][x]
                    if self.item_mode == 'targeting_wall':
                        # paredes interiores
                        if cell == 'w' and not (x == 0 or y == 0 or
                                                 x == GRID-1 or y == GRID-1):
                            valid_cells.append((x, y))
                    elif self.item_mode == 'targeting_box':
                        if cell in ('b', 'bot'):
                            valid_cells.append((x, y))
                    elif self.item_mode == 'targeting_target':
                        if cell == 'o':
                            valid_cells.append((x, y))
            # parpadeo suave
            phase = (pg.time.get_ticks() // 350) % 2
            for (x, y) in valid_cells:
                glow = pg.Surface((TILE, TILE), pg.SRCALPHA)
                alpha = 120 if phase == 0 else 70
                color = (255, 200, 80, alpha) if self.item_mode == 'targeting_wall' \
                        else (90, 220, 130, alpha) if self.item_mode == 'targeting_box' \
                        else (90, 160, 255, alpha)
                pg.draw.rect(glow, color, (0, 0, TILE, TILE), 5)
                self.window.blit(glow, (x * TILE, y * TILE))

        # HUD superior (dos filas, una para estado, otra para controles)
        hud_h = 54
        hud = pg.Surface((WINDOW_SIZE, hud_h), pg.SRCALPHA)
        hud.fill((0, 0, 0, 175))
        self.window.blit(hud, (0, 0))

        # Calcular tiempo
        elapsed_ms = pg.time.get_ticks() - self.level_start_ms
        secs = elapsed_ms // 1000
        time_str = f'{secs // 60:02d}:{secs % 60:02d}'

        # Fila 1: nivel, movs/óptimo, empujones, tiempo, monedas
        self._draw_text(f'Nv {self.current_level}/{self.total_levels}',
                        self.font_small, TEXT, (12, 6))

        # Movs con indicador de óptimo
        if self.solver_result is not None:
            opt = self.solver_result
            if self.player_moves > opt:
                movs_color = BAD       # ya superaste el óptimo
            elif self.player_moves == opt:
                movs_color = GOOD      # exactamente
            else:
                movs_color = TEXT      # todavía dentro
            movs_str = f'Movs: {self.player_moves}/{opt}'
        else:
            movs_color = TEXT
            movs_str = f'Movs: {self.player_moves}/...'
        self._draw_text(movs_str, self.font_small, movs_color, (110, 6))

        self._draw_text(f'Emp: {self.player_pushes}',
                        self.font_small, TEXT, (260, 6))
        self._draw_text(time_str, self.font_small, ACCENT, (360, 6))
        if self.is_replay:
            self._draw_text('(rejugando)', self.font_small, ACCENT,
                            (440, 6))
        # Monedas a la derecha con icono
        coins = player.get('coins', 0)
        self._draw_coin_label(coins,
                              (WINDOW_SIZE - 12 - 80, 6),
                              font=self.font_small)

        # Fila 2: controles + inventario de items
        bombs = self.data.get_item_count(self.player_name, 'bomb')
        magics = self.data.get_item_count(self.player_name, 'magic')
        ctrl_text = (f'WASD: mover · Z: deshacer · R: reiniciar · ESC: salir   '
                     f'|   B: bomba ({bombs}) · M: mágico ({magics})')
        help_surf = self.font_tiny.render(ctrl_text, True, TEXT_DIM)
        self.window.blit(help_surf,
                         ((WINDOW_SIZE - help_surf.get_width()) // 2, 32))

        # Banner de modo de item activo
        if self.item_mode != 'idle':
            msg = {
                'targeting_wall': '[B] BOMBA · haz clic en una pared (ESC para cancelar)',
                'targeting_box': '[M] PASE MÁGICO · elige una caja',
                'targeting_target': '[M] PASE MÁGICO · ahora elige un objetivo',
            }.get(self.item_mode, '')
            banner = pg.Surface((WINDOW_SIZE, 36), pg.SRCALPHA)
            banner.fill((0, 0, 0, 200))
            self.window.blit(banner, (0, WINDOW_SIZE - 36))
            self._draw_text(msg, self.font_small, ACCENT,
                            (WINDOW_SIZE//2, WINDOW_SIZE - 18), center=True)

        # Animación de explosión de bomba
        if self.explosion is not None:
            ex = self.explosion
            elapsed = pg.time.get_ticks() - ex['start_ms']
            if elapsed >= ex['duration_ms']:
                self.explosion = None
            else:
                t = elapsed / ex['duration_ms']  # 0..1
                cx, cy = ex['cx'], ex['cy']
                # Onda expansiva: 3 anillos en distintas fases
                for ring in range(3):
                    rt = max(0.0, min(1.0, t - ring * 0.12))
                    if rt <= 0:
                        continue
                    radius = int(rt * TILE * 1.6)
                    alpha = int(220 * (1 - rt))
                    if alpha <= 0 or radius <= 0:
                        continue
                    surf = pg.Surface((radius*2 + 4, radius*2 + 4), pg.SRCALPHA)
                    pg.draw.circle(surf, (255, 200, 80, alpha),
                                   (radius + 2, radius + 2), radius, 4)
                    self.window.blit(surf, (cx - radius - 2, cy - radius - 2))
                # Flash blanco en el centro (al principio)
                if t < 0.25:
                    flash_alpha = int(220 * (1 - t / 0.25))
                    flash = pg.Surface((TILE, TILE), pg.SRCALPHA)
                    pg.draw.circle(flash, (255, 255, 220, flash_alpha),
                                   (TILE//2, TILE//2), TILE//2)
                    self.window.blit(flash, (cx - TILE//2, cy - TILE//2))
                # Partículas radiales
                import math as _math
                rng = random.Random(ex['start_ms'])
                n_parts = 12
                for i in range(n_parts):
                    ang = (i / n_parts) * 2 * _math.pi + rng.random() * 0.3
                    dist = t * TILE * (1.0 + rng.random() * 0.6)
                    px = cx + _math.cos(ang) * dist
                    py = cy + _math.sin(ang) * dist
                    size = max(1, int(4 * (1 - t)))
                    col = (255, 220, 120) if rng.random() < 0.6 else (200, 100, 50)
                    pg.draw.circle(self.window, col, (int(px), int(py)), size)

    def _handle_play_key(self, event):
        """Procesa una tecla durante la partida: movimiento, deshacer,
        reiniciar, activar items y comprobar victoria."""
        # ESC: si hay modo item activo, cancelarlo; si no, salir al selector
        if event.key == pg.K_ESCAPE:
            if self.item_mode != 'idle':
                self.item_mode = 'idle'
                self.magic_pending_box = None
                return
            self.state = self.S_SELECT
            return

        # Activar items (solo si no estamos ya en un modo)
        if event.key == pg.K_b and self.item_mode == 'idle':
            if self.data.get_item_count(self.player_name, 'bomb') > 0:
                self.item_mode = 'targeting_wall'
                self.sounds.play('click')
            else:
                self.sounds.play('error')
            return
        if event.key == pg.K_m and self.item_mode == 'idle':
            if self.data.get_item_count(self.player_name, 'magic') > 0:
                self.item_mode = 'targeting_box'
                self.magic_pending_box = None
                self.sounds.play('click')
            else:
                self.sounds.play('error')
            return

        # Durante una animación de movimiento ignoramos input direccional
        # (pero R, ESC y Z funcionan: terminan la animación inmediatamente).
        if event.key in (pg.K_UP, pg.K_w, pg.K_DOWN, pg.K_s,
                         pg.K_LEFT, pg.K_a, pg.K_RIGHT, pg.K_d):
            if self._animation_active():
                # forzamos terminar animación para responder rápido
                self.animation = None
            # si estamos en modo item, ignorar direccionales
            if self.item_mode != 'idle':
                return
        if event.key in (pg.K_UP, pg.K_w):
            self._move(0, -1)
        elif event.key in (pg.K_DOWN, pg.K_s):
            self._move(0, 1)
        elif event.key in (pg.K_LEFT, pg.K_a):
            self._move(-1, 0)
        elif event.key in (pg.K_RIGHT, pg.K_d):
            self._move(1, 0)
        elif event.key == pg.K_z:
            self._undo()
            return  # deshacer no puede provocar victoria
        elif event.key == pg.K_r:
            self._load_level(self.current_level)
            return

        if self._check_win():
            # registrar tiempo (antes de que se mueva más)
            self.last_time_ms = pg.time.get_ticks() - self.level_start_ms
            # esperar a que el solver acabe brevemente para detectar óptimo
            if self.solver_thread is not None:
                self.solver_thread.join(timeout=1.0)
            prev_max = self.data.get_player(self.player_name).get('max_completed', 0)
            self._award_rewards()
            if self.current_level == self.total_levels and prev_max < self.total_levels:
                self.sounds.play('win_game')
                self.state = self.S_WIN_GAME
            else:
                self.sounds.play('win_level')
                self.state = self.S_WIN_LEVEL

    # ── Estado: NIVEL COMPLETADO ────────────────────────────────────────────
    def _draw_win_level(self):
        """Dibuja la pantalla de nivel completado."""
        self._draw_play()  # mostrar tablero detrás
        # overlay
        overlay = pg.Surface((WINDOW_SIZE, WINDOW_SIZE), pg.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        self.window.blit(overlay, (0, 0))

        self._draw_text('¡NIVEL COMPLETADO!', self.font_big, GOOD,
                        (WINDOW_SIZE//2, 130), center=True)
        self._draw_text(f'Movimientos: {self.player_moves}',
                        self.font_med, TEXT,
                        (WINDOW_SIZE//2, 184), center=True)
        # tiempo
        s = self.last_time_ms // 1000
        ms = self.last_time_ms % 1000
        time_str = f'Tiempo: {s // 60:02d}:{s % 60:02d}.{ms // 100}'
        self._draw_text(time_str, self.font_small, TEXT_DIM,
                        (WINDOW_SIZE//2, 215), center=True)
        self._draw_text(f'Empujones: {self.player_pushes}',
                        self.font_small, TEXT_DIM,
                        (WINDOW_SIZE//2, 240), center=True)
        if self.solver_result is not None:
            opt_msg = f'Óptimo: {self.solver_result} movimientos'
            opt_color = GOOD if self.player_moves == self.solver_result else TEXT_DIM
        else:
            opt_msg = 'Óptimo: calculando...'
            opt_color = TEXT_DIM
        self._draw_text(opt_msg, self.font_small, opt_color,
                        (WINDOW_SIZE//2, 265), center=True)

        # Posición en el ranking del nivel
        if self.last_position is not None:
            pos_msg = f'#{self.last_position} en el ranking de este nivel'
            pos_color = ACCENT if self.last_position <= 3 else TEXT
            self._draw_text(pos_msg, self.font_small, pos_color,
                            (WINDOW_SIZE//2, 292), center=True)

        # Recompensas
        y = 322
        for msg, val in self.last_rewards:
            color = GOOD if val > 0 else TEXT_DIM
            self._draw_text(msg, self.font_small, color,
                            (WINDOW_SIZE//2, y), center=True)
            y += 28

        player = self.data.get_player(self.player_name)
        y = max(y + 15, 470)
        lbl = self.font_med.render('Monedas totales:', True, TEXT)
        coin_text = self.font_med.render(str(player.get("coins", 0)), True, ACCENT)
        total_w = lbl.get_width() + 12 + 24 + 6 + coin_text.get_width()
        x_start = WINDOW_SIZE//2 - total_w//2
        self.window.blit(lbl, (x_start, y - lbl.get_height()//2))
        self._icon_coin(x_start + lbl.get_width() + 12 + 12, y, 12)
        self.window.blit(coin_text,
                         (x_start + lbl.get_width() + 12 + 24 + 6,
                          y - coin_text.get_height()//2))

        # Botones
        next_lvl = self.current_level + 1
        can_next = next_lvl <= self.total_levels and \
                   next_lvl <= player.get('level', 1)
        if can_next:
            self._button(pg.Rect(WINDOW_SIZE//2 - 245, 540, 160, 50),
                         'Siguiente >',
                         lambda: self._play_level(next_lvl),
                         border=GOOD, font=self.font_small)
            self._button(pg.Rect(WINDOW_SIZE//2 - 80, 540, 160, 50),
                         'Mapa', lambda: self._goto(self.S_SELECT),
                         font=self.font_small)
            self._button(pg.Rect(WINDOW_SIZE//2 + 85, 540, 160, 50),
                         'Top del nivel',
                         lambda: self._goto_level_top(self.current_level),
                         border=ACCENT, font=self.font_small)
        else:
            self._button(pg.Rect(WINDOW_SIZE//2 - 165, 540, 160, 50),
                         'Mapa', lambda: self._goto(self.S_SELECT),
                         border=GOOD, font=self.font_small)
            self._button(pg.Rect(WINDOW_SIZE//2 + 5, 540, 160, 50),
                         'Top del nivel',
                         lambda: self._goto_level_top(self.current_level),
                         border=ACCENT, font=self.font_small)

    # ── Estado: JUEGO COMPLETO ──────────────────────────────────────────────
    def _draw_win_game(self):
        """Dibuja la pantalla final (los 20 niveles superados)."""
        self._clear()
        # estrellas (preservando estado de random)
        _bk = random.getstate()
        random.seed(7)
        for _ in range(120):
            x = random.randint(0, WINDOW_SIZE)
            y = random.randint(0, WINDOW_SIZE)
            r = random.choice([1, 1, 1, 2])
            pg.draw.circle(self.window, (200, 200, 230), (x, y), r)
        random.setstate(_bk)

        # Trofeo grande dibujado
        self._icon_trophy(WINDOW_SIZE//2, 160, 50)
        self._draw_text('¡JUEGO COMPLETADO!', self.font_big, ACCENT,
                        (WINDOW_SIZE//2, 250), center=True)
        self._draw_text(f'{self.player_name}, has terminado los '
                        f'{self.total_levels} niveles',
                        self.font_small, TEXT,
                        (WINDOW_SIZE//2, 290), center=True)
        # mostrar últimas recompensas
        y = 340
        for msg, val in self.last_rewards:
            color = GOOD if val > 0 else TEXT_DIM
            self._draw_text(msg, self.font_small, color,
                            (WINDOW_SIZE//2, y), center=True)
            y += 30

        player = self.data.get_player(self.player_name)
        label = f'Monedas totales: {player.get("coins", 0)}'
        self._draw_coin_label(player.get("coins", 0),
                              (WINDOW_SIZE//2, y + 50),
                              font=self.font_med, center=True)

        self._button(pg.Rect(WINDOW_SIZE//2 - 220, 600, 200, 60),
                     'Mapa', lambda: self._goto(self.S_SELECT))
        self._button(pg.Rect(WINDOW_SIZE//2 - 0, 600, 200, 60),
                     'Tienda', lambda: self._goto_shop(),
                     border=GOOD)

    # ── Estado: TOP DEL NIVEL ───────────────────────────────────────────────
    def _goto_level_top(self, level):
        """Abre la tabla de récords del nivel indicado."""
        self.level_top_target = level
        self.state = self.S_LEVEL_TOP

    def _draw_level_top(self):
        """Dibuja la tabla de récords de un nivel concreto."""
        self._clear()
        lvl = self.level_top_target or 1

        # Header
        self._icon_trophy(WINDOW_SIZE//2 - 200, 50, 22)
        self._draw_text(f'TOP - Nivel {lvl}', self.font_big, ACCENT,
                        (WINDOW_SIZE//2 + 20, 50), center=True)

        # Botón volver: si veníamos de win_level mejor volver allí,
        # si veníamos del selector volver al selector.
        self._button(pg.Rect(20, 30, 110, 40), '< Volver',
                     lambda: self._level_top_back(),
                     font=self.font_small)

        # Mini-preview del nivel (a la derecha)
        mini = self._get_mini_board(lvl, tile_size=14)
        mw, mh = mini.get_size()
        mx = WINDOW_SIZE - mw - 20
        my = 95
        bg_rect = pg.Rect(mx - 6, my - 6, mw + 12, mh + 12)
        pg.draw.rect(self.window, BG_PANEL, bg_rect, border_radius=6)
        pg.draw.rect(self.window, PRIMARY, bg_rect, 2, border_radius=6)
        self.window.blit(mini, (mx, my))

        # Óptimo conocido para el nivel
        opt = self.data.get_min_moves(lvl)
        if opt is not None:
            self._draw_text(f'Óptimo: {opt} movs', self.font_small, TEXT_DIM,
                            (mx - 10, my + mh + 10))

        # Lista del top
        top = self.data.get_level_top(lvl, limit=10)
        if not top:
            self._draw_text('Aún nadie ha completado este nivel.',
                            self.font_small, TEXT_DIM,
                            (40, 180))
            return

        # Cabeceras
        hdr_y = 110
        self._draw_text('#', self.font_small, TEXT_DIM, (30, hdr_y))
        self._draw_text('Jugador', self.font_small, TEXT_DIM, (75, hdr_y))
        self._draw_text('Movs', self.font_small, TEXT_DIM, (260, hdr_y))
        self._draw_text('Emp.', self.font_small, TEXT_DIM, (330, hdr_y))
        self._draw_text('Tiempo', self.font_small, TEXT_DIM, (400, hdr_y))

        row_h = 44
        medal_colors = [(255, 215, 0), (200, 200, 215), (200, 130, 60)]
        for i, r in enumerate(top):
            y = 145 + i * row_h
            rect = pg.Rect(20, y, 480, row_h - 6)
            bg = (40, 40, 55) if i % 2 == 0 else (32, 32, 46)
            if r.get('name') == self.player_name:
                bg = (60, 80, 110)
            pg.draw.rect(self.window, bg, rect, border_radius=6)

            mid = y + (row_h - 6)//2
            if i < 3:
                pg.draw.circle(self.window, medal_colors[i], (38, mid), 12)
                pg.draw.circle(self.window, (80, 60, 20), (38, mid), 12, 2)
                num_surf = self.font_tiny.render(str(i+1), True, (40, 30, 10))
                self.window.blit(num_surf, num_surf.get_rect(center=(38, mid)))
            else:
                self._draw_text(str(i+1), self.font_small, TEXT,
                                (38, mid), center=True)

            self._draw_text(r.get('name', '?'), self.font_med, TEXT,
                            (75, mid - 13))
            self._draw_text(str(r.get('moves', '?')), self.font_med, TEXT,
                            (265, mid - 13))
            self._draw_text(str(r.get('pushes', '-')), self.font_small, TEXT_DIM,
                            (335, mid - 10))
            tms = r.get('time_ms', 0)
            s = tms // 1000
            tstr = f'{s // 60:02d}:{s % 60:02d}.{(tms % 1000) // 100}'
            self._draw_text(tstr, self.font_small, TEXT_DIM,
                            (405, mid - 10))

    def _level_top_back(self):
        """Vuelve atrás desde la tabla de récords del nivel."""
        # heurística: si tenemos un nivel cargado lo más útil es volver al
        # mapa (el flujo más común es: selector -> level_top -> selector,
        # o win_level -> level_top -> selector)
        self._goto(self.S_SELECT)

    # ── Estado: AJUSTES ─────────────────────────────────────────────────────
    def _draw_settings(self):
        """Dibuja la pantalla de ajustes de audio."""
        self._clear()
        # Header
        self._icon_gear(WINDOW_SIZE//2 - 130, 50, 18)
        self._draw_text('AJUSTES', self.font_big, TEXT,
                        (WINDOW_SIZE//2 + 20, 50), center=True)
        self._button(pg.Rect(20, 30, 110, 40), '< Menú',
                     lambda: self._goto(self.S_MENU), font=self.font_small)

        if not self.sounds.ok:
            self._draw_text('(Audio no disponible en este sistema)',
                            self.font_small, BAD,
                            (WINDOW_SIZE//2, 130), center=True)

        # Sección: Música
        y = 150
        self._draw_text('Música', self.font_med, TEXT, (60, y))
        # icono altavoz a la izquierda
        self._icon_speaker(40, y + 16, 14,
                           muted=not self.sounds.music_enabled)
        # Toggle ON/OFF
        toggle_rect = pg.Rect(WINDOW_SIZE - 180, y - 4, 130, 40)
        is_on = self.sounds.music_enabled
        self._button(toggle_rect,
                     'ENCENDIDO' if is_on else 'APAGADO',
                     lambda: self._toggle_music(),
                     border=GOOD if is_on else BAD,
                     font=self.font_small)

        # Slider de volumen música
        y += 60
        self._draw_text(f'Volumen: {int(self.sounds.music_volume*100)}%',
                        self.font_small, TEXT_DIM, (60, y))
        self._volume_slider(60, y + 28, WINDOW_SIZE - 120,
                            self.sounds.music_volume,
                            self._set_music_vol,
                            enabled=is_on)

        # Sección: Efectos
        y += 90
        self._draw_text('Efectos de sonido', self.font_med, TEXT, (60, y))
        self._icon_speaker(40, y + 16, 14,
                           muted=not self.sounds.sfx_enabled)
        toggle_rect = pg.Rect(WINDOW_SIZE - 180, y - 4, 130, 40)
        is_on_sfx = self.sounds.sfx_enabled
        self._button(toggle_rect,
                     'ENCENDIDO' if is_on_sfx else 'APAGADO',
                     lambda: self._toggle_sfx(),
                     border=GOOD if is_on_sfx else BAD,
                     font=self.font_small)

        y += 60
        self._draw_text(f'Volumen: {int(self.sounds.sfx_volume*100)}%',
                        self.font_small, TEXT_DIM, (60, y))
        self._volume_slider(60, y + 28, WINDOW_SIZE - 120,
                            self.sounds.sfx_volume,
                            self._set_sfx_vol,
                            enabled=is_on_sfx)

        # Botón Probar (reproduce un click + step)
        y += 70
        self._button(pg.Rect(WINDOW_SIZE//2 - 110, y, 220, 50),
                     'Probar sonido',
                     lambda: self._test_sounds(),
                     font=self.font_small, border=ACCENT)

    def _volume_slider(self, x, y, w, value, on_change, *, enabled=True):
        """Slider horizontal con 11 pasos clickables (0%, 10%, ..., 100%).
        Más sencillo que arrastrar y funciona mejor con eventos discretos."""
        h = 28
        track = pg.Rect(x, y, w, h)
        color_bg = (40, 40, 55) if enabled else (28, 28, 38)
        color_fg = PRIMARY if enabled else LOCKED
        pg.draw.rect(self.window, color_bg, track, border_radius=h//2)
        # relleno
        fill_w = int(w * value)
        if fill_w > 4:
            fill_rect = pg.Rect(x, y, fill_w, h)
            pg.draw.rect(self.window, color_fg, fill_rect, border_radius=h//2)
        pg.draw.rect(self.window, (90, 100, 130), track, 2, border_radius=h//2)

        # Marcas de cada 10%
        for i in range(11):
            mx = x + int(w * i / 10)
            pg.draw.line(self.window, (160, 160, 175),
                         (mx, y + h - 4), (mx, y + h), 1)

        # área clickable
        if enabled:
            self.buttons.append((track,
                                 lambda r=track, ox=x, ww=w: self._slider_click(
                                     r, ox, ww, on_change)))

    def _slider_click(self, rect, ox, w, on_change):
        """Traduce un clic sobre un slider a un valor 0.0-1.0 (en pasos de 0.1)."""
        # calcular posición del click (relativa)
        mx = pg.mouse.get_pos()[0]
        v = (mx - ox) / w
        v = max(0.0, min(1.0, v))
        # cuantizar a múltiplos de 0.1
        v = round(v * 10) / 10
        on_change(v)

    def _toggle_music(self):
        """Activa/desactiva la música y guarda el ajuste."""
        new = not self.sounds.music_enabled
        self.sounds.set_music_enabled(new)
        self.data.set_setting('music_enabled', new)
        # si la activamos en menú, reanudar loop
        if new and self.state == self.S_SETTINGS:
            # no rearrancamos aquí; sólo arranca al volver al menú
            pass

    def _toggle_sfx(self):
        """Activa/desactiva los efectos de sonido y guarda el ajuste."""
        new = not self.sounds.sfx_enabled
        self.sounds.set_sfx_enabled(new)
        self.data.set_setting('sfx_enabled', new)

    def _set_music_vol(self, v):
        """Fija el volumen de música y lo persiste."""
        self.sounds.set_music_volume(v)
        self.data.set_setting('music_volume', v)

    def _set_sfx_vol(self, v):
        """Fija el volumen de efectos y lo persiste."""
        self.sounds.set_sfx_volume(v)
        self.data.set_setting('sfx_volume', v)

    def _test_sounds(self):
        """Reproduce un par de sonidos de muestra (botón "Probar")."""
        # reproduce un par para que se oiga el ajuste
        self.sounds.play('click')
        # step un pelín después
        if self.sounds.ok:
            pg.time.set_timer(pg.USEREVENT + 1, 200, loops=1)

    # ── Estado: RANKING ─────────────────────────────────────────────────────
    def _draw_rank(self):
        """Dibuja el ranking global de jugadores (estado S_RANK)."""
        self._clear()
        # título con trofeo dibujado
        self._icon_trophy(WINDOW_SIZE//2 - 130, 50, 22)
        self._draw_text('RANKING', self.font_big, ACCENT,
                        (WINDOW_SIZE//2 + 20, 50), center=True)
        self._button(pg.Rect(20, 30, 110, 40), '< Menú',
                     lambda: self._goto(self.S_MENU), font=self.font_small)

        # cabeceras
        hdr_y = 110
        self._draw_text('#', self.font_small, TEXT_DIM, (40, hdr_y))
        self._draw_text('Jugador', self.font_small, TEXT_DIM, (90, hdr_y))
        self._draw_text('Nivel', self.font_small, TEXT_DIM, (340, hdr_y))
        self._draw_text('Compl.', self.font_small, TEXT_DIM, (430, hdr_y))
        self._draw_text('Monedas', self.font_small, TEXT_DIM, (550, hdr_y))

        ranking = self.data.ranking()
        if not ranking:
            self._draw_text('(aún no hay jugadores)',
                            self.font_med, TEXT_DIM,
                            (WINDOW_SIZE//2, WINDOW_SIZE//2), center=True)
            return

        row_h = 52
        visible_rows = (WINDOW_SIZE - 200) // row_h
        max_scroll = max(0, len(ranking) - visible_rows)
        self.rank_scroll = max(0, min(self.rank_scroll, max_scroll))

        # colores medalla: oro / plata / bronce
        medal_colors = [(255, 215, 0), (200, 200, 215), (200, 130, 60)]

        start_y = 145
        for i, (name, info) in enumerate(ranking[self.rank_scroll:
                                                self.rank_scroll + visible_rows]):
            global_i = i + self.rank_scroll
            y = start_y + i * row_h
            rect = pg.Rect(20, y, WINDOW_SIZE - 40, row_h - 6)
            bg = (40, 40, 55) if global_i % 2 == 0 else (32, 32, 46)
            if name == self.player_name:
                bg = (60, 80, 110)
            pg.draw.rect(self.window, bg, rect, border_radius=8)

            row_mid_y = y + (row_h - 6)//2

            # número de posición / disco medalla
            if global_i < 3:
                pg.draw.circle(self.window, medal_colors[global_i],
                               (45, row_mid_y), 14)
                pg.draw.circle(self.window, (80, 60, 20),
                               (45, row_mid_y), 14, 2)
                pos_text = self.font_small.render(str(global_i + 1), True, (40, 30, 10))
                self.window.blit(pos_text, pos_text.get_rect(center=(45, row_mid_y)))
            else:
                self._draw_text(str(global_i + 1), self.font_med, TEXT,
                                (45, row_mid_y), center=True)

            self._draw_text(name, self.font_med, TEXT,
                            (90, row_mid_y - 13))
            # Nivel a mostrar: cap a total_levels (porque level puede ser 21)
            display_lvl = min(info.get('level', 1), self.total_levels)
            self._draw_text(str(display_lvl),
                            self.font_med, TEXT,
                            (360, row_mid_y - 13))
            self._draw_text(f"{info.get('max_completed', 0)}/{self.total_levels}",
                            self.font_med, TEXT,
                            (440, row_mid_y - 13))
            # Moneda con icono
            self._draw_coin_label(info.get('coins', 0),
                                  (560, row_mid_y - 13),
                                  font=self.font_med)

            # Botón papelera (no se puede borrar al jugador "logueado"
            # mientras esté jugando — pero sí desde el menú)
            trash_rect = pg.Rect(rect.right - 42, y + 4, 36, row_h - 14)
            mouse = pg.mouse.get_pos()
            hovered = trash_rect.collidepoint(mouse)
            if hovered:
                pg.draw.rect(self.window, BAD, trash_rect, border_radius=6)
            else:
                pg.draw.rect(self.window, (60, 30, 30), trash_rect, border_radius=6)
                pg.draw.rect(self.window, BAD, trash_rect, 1, border_radius=6)
            self._icon_trash(trash_rect.centerx, trash_rect.centery, 9)
            self.buttons.append((trash_rect,
                                 lambda n=name: self._request_delete(n)))

        if max_scroll > 0:
            self._draw_text(f'^v para desplazar  ({self.rank_scroll+1}-'
                            f'{min(self.rank_scroll + visible_rows, len(ranking))}'
                            f'/{len(ranking)})',
                            self.font_tiny, TEXT_DIM,
                            (WINDOW_SIZE//2, WINDOW_SIZE - 25), center=True)

        # Diálogo de confirmación
        if self.delete_confirm:
            self._draw_delete_confirm()

    def _request_delete(self, name):
        """Abre el diálogo de confirmación para borrar un jugador."""
        self.delete_confirm = name

    def _draw_delete_confirm(self):
        """Dibuja el modal de confirmación de borrado."""
        # Modal centrado
        overlay = pg.Surface((WINDOW_SIZE, WINDOW_SIZE), pg.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.window.blit(overlay, (0, 0))

        dlg = pg.Rect(WINDOW_SIZE//2 - 220, WINDOW_SIZE//2 - 110, 440, 220)
        pg.draw.rect(self.window, BG_PANEL, dlg, border_radius=12)
        pg.draw.rect(self.window, BAD, dlg, 3, border_radius=12)

        self._draw_text('¿Borrar jugador?', self.font_med, TEXT,
                        (WINDOW_SIZE//2, dlg.y + 40), center=True)
        self._draw_text(f'"{self.delete_confirm}"', self.font_big, BAD,
                        (WINDOW_SIZE//2, dlg.y + 90), center=True)
        self._draw_text('Esta acción no se puede deshacer.',
                        self.font_small, TEXT_DIM,
                        (WINDOW_SIZE//2, dlg.y + 130), center=True)

        # Botones
        self._button(pg.Rect(dlg.x + 30, dlg.bottom - 70, 170, 50),
                     'Cancelar',
                     lambda: self._cancel_delete(),
                     font=self.font_small)
        self._button(pg.Rect(dlg.right - 200, dlg.bottom - 70, 170, 50),
                     'Sí, borrar',
                     lambda: self._confirm_delete(),
                     font=self.font_small, border=BAD)

    def _cancel_delete(self):
        """Cierra el modal de borrado sin borrar nada."""
        self.delete_confirm = None

    def _confirm_delete(self):
        """Borra definitivamente el jugador y cierra el modal."""
        name = self.delete_confirm
        self.delete_confirm = None
        if name is None:
            return
        # Si era el jugador activo, lo deslogueamos
        if name == self.player_name:
            self.player_name = None
        self.data.delete_player(name)

    # ── Estado: TIENDA ──────────────────────────────────────────────────────
    def _draw_shop(self):
        """Dibuja la tienda con sus tres pestañas (estado S_SHOP)."""
        self._clear()
        # Título con icono carrito
        self._icon_cart(WINDOW_SIZE//2 - 90, 50, 22)
        self._draw_text('TIENDA', self.font_big, GOOD,
                        (WINDOW_SIZE//2 + 30, 45), center=True)

        if self.player_name is None:
            self._draw_text('Crea un jugador primero', self.font_med, BAD,
                            (WINDOW_SIZE//2, WINDOW_SIZE//2), center=True)
            self._button(pg.Rect(20, 20, 110, 40), '< Menú',
                         lambda: self._goto(self.S_MENU), font=self.font_small)
            return

        player = self.data.get_player(self.player_name)
        coins = player.get('coins', 0)

        self._button(pg.Rect(20, 20, 110, 40), '< Menú',
                     lambda: self._goto(self.S_MENU), font=self.font_small)
        # Monedas a la derecha con icono vectorial
        self._draw_coin_label(coins, (WINDOW_SIZE - 110, 26),
                              font=self.font_med)

        # Tabs
        tab_y = 100
        tab_h = 42
        tab_defs = [
            ('char', 'Personajes'),
            ('box', 'Cajas'),
            ('item', 'Items'),
        ]
        tab_w = (WINDOW_SIZE - 60) // 3
        for i, (key, label) in enumerate(tab_defs):
            tx = 30 + i * tab_w
            rect = pg.Rect(tx, tab_y, tab_w - 4, tab_h)
            is_active = self.shop_tab == key
            self._button(rect, label,
                         lambda k=key: self._set_shop_tab(k),
                         border=ACCENT if is_active else (90, 100, 130),
                         bg=(50, 50, 70) if is_active else BG_PANEL,
                         fg=ACCENT if is_active else TEXT,
                         font=self.font_small)

        # Contenido según tab
        if self.shop_tab == 'char':
            self._draw_shop_grid(CHARACTER_SKINS, 'char', player)
            hint = 'Comprar = monedas  ·  Botón verde = elegir'
        elif self.shop_tab == 'box':
            self._draw_shop_grid(BOX_SKINS, 'box', player)
            hint = 'Comprar = monedas  ·  Botón verde = elegir'
        else:  # item
            self._draw_shop_items(player)
            hint = 'Items consumibles. Úsalos en partida con B (bomba) o M (mágico)'

        self._draw_text(hint,
                        self.font_small, TEXT_DIM,
                        (WINDOW_SIZE//2, WINDOW_SIZE - 30), center=True)

    def _set_shop_tab(self, tab):
        """Cambia la pestaña activa de la tienda."""
        self.shop_tab = tab

    def _draw_shop_grid(self, registry, kind, player):
        """Grid 5xN de skins (personajes o cajas)."""
        x0, y0 = 30, 165
        cell_w, cell_h = 130, 180
        gap = 10
        # Una fila con todos (hasta 5)
        for i, (key, info) in enumerate(registry.items()):
            x = x0 + i * (cell_w + gap)
            self._draw_shop_item(kind, key, info, x, y0, cell_w, cell_h, player)

    def _draw_shop_items(self, player):
        """Grid de items consumibles."""
        x0, y0 = 60, 175
        cell_w, cell_h = 280, 220
        gap = 20
        for i, (key, info) in enumerate(ITEMS.items()):
            x = x0 + i * (cell_w + gap)
            self._draw_shop_item_consumable(key, info, x, y0,
                                            cell_w, cell_h, player)

    def _draw_shop_item_consumable(self, key, info, x, y, w, h, player):
        """Dibuja la tarjeta de un item consumible en la tienda."""
        rect = pg.Rect(x, y, w, h)
        owned = player.get('items', {}).get(key, 0)
        coins = player.get('coins', 0)
        price = info['price']
        can_buy = coins >= price

        bg = BG_PANEL
        border = PRIMARY if owned > 0 else (90, 100, 130)
        pg.draw.rect(self.window, bg, rect, border_radius=12)
        pg.draw.rect(self.window, border, rect, 2, border_radius=12)

        # icono
        icon_size = 100
        icon = info['maker'](icon_size)
        self.window.blit(icon, (x + (w - icon_size)//2, y + 10))

        # nombre
        self._draw_text(info['name'], self.font_med, TEXT,
                        (x + w//2, y + 125), center=True)
        # descripción
        self._draw_text(info['desc'], self.font_tiny, TEXT_DIM,
                        (x + w//2, y + 150), center=True)
        # cantidad poseída
        self._draw_text(f'Tienes: {owned}', self.font_small,
                        ACCENT if owned > 0 else TEXT_DIM,
                        (x + 20, y + h - 60))
        # precio
        self._draw_coin_label(price, (x + w - 110, y + h - 65),
                              font=self.font_small)
        # botón comprar
        btn = pg.Rect(x + 20, y + h - 36, w - 40, 28)
        self._button(btn,
                     'Comprar' if can_buy else 'Sin saldo',
                     lambda k=key, p=price: self._buy_item(k, p),
                     font=self.font_small,
                     border=GOOD if can_buy else BAD,
                     disabled=not can_buy)

    def _buy_item(self, key, price):
        """Compra una unidad de un item si hay monedas suficientes."""
        player = self.data.get_player(self.player_name)
        if player.get('coins', 0) < price:
            self.sounds.play('error')
            return
        new_coins = player['coins'] - price
        self.data.update_player(self.player_name, coins=new_coins)
        self.data.add_item(self.player_name, key, 1)
        self.sounds.play('click')

    def _draw_shop_item(self, kind, key, info, x, y, w, h, player):
        """Dibuja la tarjeta de un skin (personaje o caja) en la tienda."""
        rect = pg.Rect(x, y, w, h)
        unlocked_key = 'unlocked_chars' if kind == 'char' else 'unlocked_boxes'
        current_key = 'character' if kind == 'char' else 'box_skin'
        is_unlocked = key in player.get(unlocked_key, [])
        is_selected = player.get(current_key) == key

        if is_selected:
            bg = (40, 80, 60)
            border = GOOD
        elif is_unlocked:
            bg = BG_PANEL
            border = PRIMARY
        else:
            bg = (35, 35, 45)
            border = (80, 80, 95)

        pg.draw.rect(self.window, bg, rect, border_radius=12)
        pg.draw.rect(self.window, border, rect, 2, border_radius=12)

        # preview
        preview_size = 80
        preview = info['maker'](preview_size)
        self.window.blit(preview, (x + (w - preview_size)//2, y + 10))

        # nombre
        self._draw_text(info['name'], self.font_small, TEXT,
                        (x + w//2, y + 100), center=True)

        # estado / precio / botón
        if is_selected:
            cx = x + w//2 - 32
            cy = y + 125
            self._icon_check(cx, cy, 7)
            self._draw_text('En uso', self.font_tiny, GOOD,
                            (x + w//2 + 8, cy), center=True)
        elif is_unlocked:
            btn = pg.Rect(x + 10, y + 130, w - 20, 30)
            self._button(btn, 'Elegir',
                         lambda kn=kind, k=key: self._select_skin(kn, k),
                         font=self.font_small, border=GOOD)
        else:
            price = info['price']
            self._draw_coin_label(price, (x + w//2, y + 125),
                                  font=self.font_small, center=True)
            btn = pg.Rect(x + 10, y + 130, w - 20, 30)
            can_buy = player.get('coins', 0) >= price
            self._button(btn, 'Comprar' if can_buy else 'Sin saldo',
                         lambda kn=kind, k=key, p=price: self._buy_skin(kn, k, p),
                         font=self.font_small,
                         border=GOOD if can_buy else BAD,
                         disabled=not can_buy)

    def _select_skin(self, kind, key):
        """Equipa un skin ya comprado por el jugador."""
        if kind == 'char':
            self.data.update_player(self.player_name, character=key)
        else:
            self.data.update_player(self.player_name, box_skin=key)
        # invalidar cache (no estrictamente necesario, mismo key)

    def _buy_skin(self, kind, key, price):
        """Compra un skin si hay monedas suficientes."""
        player = self.data.get_player(self.player_name)
        if player.get('coins', 0) < price:
            return
        new_coins = player['coins'] - price
        if kind == 'char':
            unlocked = list(player.get('unlocked_chars', []))
            if key not in unlocked:
                unlocked.append(key)
            self.data.update_player(self.player_name,
                                    coins=new_coins,
                                    unlocked_chars=unlocked,
                                    character=key)
        else:
            unlocked = list(player.get('unlocked_boxes', []))
            if key not in unlocked:
                unlocked.append(key)
            self.data.update_player(self.player_name,
                                    coins=new_coins,
                                    unlocked_boxes=unlocked,
                                    box_skin=key)

    # ── Manejo de eventos ───────────────────────────────────────────────────
    def _handle_event(self, event):
        """Procesa UN evento de pygame (teclado, ratón, cierre de ventana)
        según el estado actual del juego."""
        if event.type == pg.QUIT:
            self._quit()

        # USEREVENT+1: segundo sonido del botón "Probar" en ajustes
        if event.type == pg.USEREVENT + 1:
            self.sounds.play('step')
            return

        # Si hay modal de confirmación abierto, solo aceptamos sus botones,
        # Escape para cancelar y click fuera para cancelar.
        modal_active = (self.state == self.S_RANK and self.delete_confirm)

        # Manejo común de click en botones (recolectados durante draw)
        if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
            if modal_active:
                # Sólo botones DENTRO del modal son válidos
                dlg = pg.Rect(WINDOW_SIZE//2 - 220, WINDOW_SIZE//2 - 110, 440, 220)
                if not dlg.collidepoint(event.pos):
                    self._cancel_delete()
                    return
                # buscar el botón del modal que coincida
                for rect, cb in self.buttons:
                    if dlg.contains(rect) and rect.collidepoint(event.pos):
                        self.sounds.play('click')
                        cb()
                        return
                return

            # En estado S_PLAY con un item activo, el click va al tablero
            if self.state == self.S_PLAY and self.item_mode != 'idle':
                cell = self._click_to_cell(event.pos)
                if cell is not None:
                    self._try_use_item_at(cell)
                    return

            for rect, cb in self.buttons:
                if rect.collidepoint(event.pos):
                    self.sounds.play('click')
                    cb()
                    return

        if event.type == pg.MOUSEBUTTONDOWN and event.button in (4, 5):
            delta = -1 if event.button == 4 else 1
            if self.state == self.S_RANK and not modal_active:
                self.rank_scroll = max(0, self.rank_scroll + delta)
            elif self.state == self.S_SHOP:
                self.shop_scroll = max(0, self.shop_scroll + delta)

        if event.type == pg.KEYDOWN:
            # Modal de borrado: Escape cancela, Enter confirma
            if modal_active:
                if event.key == pg.K_ESCAPE:
                    self._cancel_delete()
                elif event.key == pg.K_RETURN:
                    self._confirm_delete()
                return

            if self.state == self.S_NAME:
                if event.key == pg.K_RETURN:
                    self._submit_name()
                elif event.key == pg.K_BACKSPACE:
                    self.name_input = self.name_input[:-1]
                    self.name_error = ''
                elif event.key == pg.K_ESCAPE:
                    self._goto(self.S_MENU)
                else:
                    ch = event.unicode
                    if ch and ch.isprintable() and len(self.name_input) < 12:
                        # filtrar caracteres permitidos
                        if ch.isalnum() or ch in (' ', '_', '-', '.'):
                            self.name_input += ch
                            self.name_error = ''
            elif self.state == self.S_PLAY:
                self._handle_play_key(event)
            elif self.state == self.S_RANK:
                if event.key == pg.K_UP:
                    self.rank_scroll = max(0, self.rank_scroll - 1)
                elif event.key == pg.K_DOWN:
                    self.rank_scroll += 1
                elif event.key == pg.K_ESCAPE:
                    self._goto(self.S_MENU)
            elif self.state == self.S_SHOP:
                if event.key == pg.K_ESCAPE:
                    self._goto(self.S_MENU)
            elif self.state == self.S_SELECT:
                if event.key == pg.K_ESCAPE:
                    self._goto(self.S_MENU)
            elif self.state == self.S_LEVEL_TOP:
                if event.key == pg.K_ESCAPE:
                    self._level_top_back()
            elif self.state == self.S_SETTINGS:
                if event.key == pg.K_ESCAPE:
                    self._goto(self.S_MENU)
            elif self.state == self.S_MENU:
                if event.key == pg.K_ESCAPE:
                    self._quit()
            elif self.state in (self.S_WIN_LEVEL, self.S_WIN_GAME):
                if event.key == pg.K_ESCAPE:
                    self._goto(self.S_SELECT)
                elif event.key == pg.K_RETURN and self.state == self.S_WIN_LEVEL:
                    nxt = self.current_level + 1
                    player = self.data.get_player(self.player_name)
                    if nxt <= self.total_levels and nxt <= player.get('level', 1):
                        self._play_level(nxt)
                    else:
                        self._goto(self.S_SELECT)

    # ── Loop principal ──────────────────────────────────────────────────────
    def run(self):
        """Bucle principal del juego. En cada iteración: dibuja la pantalla
        del estado actual, procesa los eventos y refresca a 60 FPS."""
        clock = pg.time.Clock()
        while True:
            # Dibujar según estado (esto también registra los botones)
            if self.state == self.S_MENU:
                self._draw_menu()
            elif self.state == self.S_NAME:
                self._draw_name()
            elif self.state == self.S_SELECT:
                self._draw_select()
            elif self.state == self.S_PLAY:
                self._draw_play()
            elif self.state == self.S_WIN_LEVEL:
                self._draw_win_level()
            elif self.state == self.S_WIN_GAME:
                self._draw_win_game()
            elif self.state == self.S_RANK:
                self._draw_rank()
            elif self.state == self.S_SHOP:
                self._draw_shop()
            elif self.state == self.S_LEVEL_TOP:
                self._draw_level_top()
            elif self.state == self.S_SETTINGS:
                self._draw_settings()

            for event in pg.event.get():
                self._handle_event(event)

            pg.display.flip()
            clock.tick(60)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    SokobanGame().run()
