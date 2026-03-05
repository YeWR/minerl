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


def main():
    global pygame
    import pygame as pg
    pygame = pg

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='MineRLPlanChopTreeCaveHills-v0')
    args = parser.parse_args()

    env = gym.make(args.env)
    print(f"Environment: {args.env}")
    print(f"Action space keys: {list(env.action_space.spaces.keys())}")

    obs = env.reset()
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

        if reward != 0:
            print(f"[Step {step_count}] reward={reward:.1f}  total={total_reward:.1f}")

        pov = obs['pov']
        surface = pygame.surfarray.make_surface(np.transpose(pov, (1, 0, 2)))
        surface = pygame.transform.scale(surface, (display_w, display_h))
        screen.blit(surface, (0, 0))

        info_text = f"Step: {step_count}  Reward: {total_reward:.1f}"
        font = pygame.font.SysFont(None, 28)
        text_surface = font.render(info_text, True, (255, 255, 0))
        screen.blit(text_surface, (10, 10))

        pygame.display.flip()
        clock.tick(20)

    print(f"\nEpisode finished at step {step_count}, total reward: {total_reward:.1f}")
    if total_reward >= 1.0:
        print("SUCCESS!")
    else:
        print("FAILED (timeout or quit)")

    env.close()
    pygame.quit()


if __name__ == '__main__':
    main()
