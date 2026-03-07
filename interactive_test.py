"""
Interactive test for planning tasks.
Usage:
    python interactive_test.py --env MineRLPlanChopTreeCaveHills-v0

Controls (pygame window must be focused):
    WASD        move
    Space       jump
    LShift      sneak
    LCtrl       sprint
    F           attack (hold to break blocks)
    G           use (right click)
    Arrow keys  camera (look around)
    1-9         hotbar slot
    ESC         quit episode
"""

import argparse
import sys
import time

import gym
import numpy as np
import minerl

pygame = None

CAMERA_SPEED = 5.0

KEY_TO_ACTION = {
    'w': 'forward',
    's': 'back',
    'a': 'left',
    'd': 'right',
    'space': 'jump',
    'lshift': 'sneak',
    'lctrl': 'sprint',
    'f': 'attack',
    'g': 'use',
    'e': 'inventory',
    'q': 'drop',
}

HOTBAR_KEYS = {
    str(i): f'hotbar.{i}' for i in range(1, 10)
}


def get_pygame_key_name(key):
    name = pygame.key.name(key)
    return name.lower().replace(' ', '')


def build_action(env, pressed_keys, camera_delta):
    action = {k: env.action_space[k].sample() * 0 for k in env.action_space.spaces}
    action['camera'] = np.array([0.0, 0.0])

    for key in pressed_keys:
        name = get_pygame_key_name(key)
        if name in KEY_TO_ACTION:
            ac_name = KEY_TO_ACTION[name]
            if ac_name in action:
                action[ac_name] = 1
        if name in HOTBAR_KEYS:
            ac_name = HOTBAR_KEYS[name]
            if ac_name in action:
                action[ac_name] = 1

    action['camera'] = np.array(camera_delta, dtype=np.float32)

    if pygame.K_ESCAPE in pressed_keys and 'ESC' in action:
        action['ESC'] = 1

    return action


def dump_obs(obs, label=""):
    """Print detailed obs structure."""
    print(f"\n=== OBS DUMP {label} ===")
    for k, v in obs.items():
        if k == 'pov':
            print(f"  pov: shape={v.shape} dtype={v.dtype}")
        elif isinstance(v, dict):
            nonzero = {ik: int(iv) for ik, iv in v.items() if int(iv) != 0}
            print(f"  {k}: {nonzero if nonzero else '{all zero}'}")
        elif isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape} val={v}")
        else:
            print(f"  {k}: {v}")
    print("=== END DUMP ===\n")


def get_inventory_nonzero(obs):
    if 'inventory' in obs and isinstance(obs['inventory'], dict):
        return {k: int(v) for k, v in obs['inventory'].items() if int(v) != 0}
    return None


def main():
    global pygame
    import pygame as pg
    pygame = pg

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='MineRLPlanChopTreeCaveHills-v0')
    args = parser.parse_args()

    env = gym.make(args.env)
    print(f"=== Environment: {args.env} ===")
    print(f"Action space keys: {list(env.action_space.spaces.keys())}")
    print(f"Observation space keys: {list(env.observation_space.spaces.keys())}")

    obs = env.reset()
    dump_obs(obs, "RESET")

    pov = obs['pov']
    h, w = pov.shape[:2]

    scale = max(1, 720 // h)
    display_w, display_h = w * scale, h * scale

    pygame.init()
    screen = pygame.display.set_mode((display_w, display_h))
    pygame.display.set_caption(f"MineRL Interactive - {args.env}")
    clock = pygame.time.Clock()

    done = False
    total_reward = 0.0
    step_count = 0
    success = False
    prev_inv = get_inventory_nonzero(obs)

    print("Waiting for actions... (focus the pygame window)")

    while not done:
        camera_delta = [0.0, 0.0]
        pressed_keys = pygame.key.get_pressed()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        held_keys = set()
        for i in range(len(pressed_keys)):
            if pressed_keys[i]:
                held_keys.add(i)

        if pressed_keys[pygame.K_UP]:
            camera_delta[0] = -CAMERA_SPEED
        if pressed_keys[pygame.K_DOWN]:
            camera_delta[0] = CAMERA_SPEED
        if pressed_keys[pygame.K_LEFT]:
            camera_delta[1] = -CAMERA_SPEED
        if pressed_keys[pygame.K_RIGHT]:
            camera_delta[1] = CAMERA_SPEED

        action = build_action(env, held_keys, camera_delta)
        obs, reward, done, info = env.step(action)

        total_reward += reward
        step_count += 1

        cur_inv = get_inventory_nonzero(obs)
        if cur_inv != prev_inv:
            print(f"[Step {step_count}] INVENTORY CHANGED: {cur_inv}  reward={reward:.2f}")
            prev_inv = cur_inv

        if reward != 0:
            print(f">>> [Step {step_count}] REWARD = {reward:.2f}  (total = {total_reward:.2f})")
            if total_reward >= 1.0:
                success = True
                print("========== TASK COMPLETE! ==========")
                done = True

        pov = obs['pov']
        surface = pygame.surfarray.make_surface(np.transpose(pov, (1, 0, 2)))
        surface = pygame.transform.scale(surface, (display_w, display_h))
        screen.blit(surface, (0, 0))

        font = pygame.font.SysFont(None, 28)
        status = "PLAYING"
        color = (255, 255, 0)
        if success:
            status = "SUCCESS!"
            color = (0, 255, 0)
        info_text = f"Step: {step_count}  Reward: {total_reward:.1f}  [{status}]"
        text_surface = font.render(info_text, True, color)
        screen.blit(text_surface, (10, 10))

        pygame.display.flip()
        clock.tick(20)

    print(f"\n{'='*40}")
    print(f"Episode finished at step {step_count}")
    print(f"Total reward: {total_reward:.2f}")
    if success:
        print("Result: SUCCESS")
    else:
        print("Result: FAILED (timeout / quit)")
    print(f"{'='*40}")

    env.close()
    pygame.quit()


if __name__ == '__main__':
    main()
