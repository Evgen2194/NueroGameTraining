import torch
from agent import Agent
from memory import ReplayMemory
from ui import UI
import config
import utils
import time
import cv2
import atexit

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = Agent(device)
    replay_memory = ReplayMemory()
    ui = UI()

    agent.memory = replay_memory

    # Load saved data
    agent.load_model()
    replay_memory.load()

    # Register exit handler
    atexit.register(agent.save_model)
    atexit.register(replay_memory.save)

    print("Welcome to Project Agent!")
    print("Please select the game area.")
    ui.select_game_area()

    stacked_frames = None
    mission_buffer = []
    mission_active = False

    # Main loop
    while True:
        # 1. Capture screen
        frame = utils.capture_screen(ui.game_area)

        # 2. Preprocess frames
        processed_frame = utils.preprocess_frame(frame, config.TARGET_RESOLUTION)
        stacked_frames = utils.stack_frames(stacked_frames, processed_frame, config.FRAME_STACK_SIZE)

        # 3. Get action from agent
        action_type, action_params = agent.get_action(stacked_frames)
        action = (action_type, action_params)

        # 4. Display suggestion and get feedback
        ui.display_action_suggestion(frame, action)
        feedback = ui.get_user_feedback()
        cv2.destroyWindow("Agent Suggestion")


        # 5. Store experience in memory
        if feedback == "start_mission":
            mission_active = True
            mission_buffer = []
            print("Mission started!")
        elif feedback == "mission_win" or feedback == "mission_loss":
            if mission_active:
                strategic_reward = config.REWARD_STRATEGIC_WIN if feedback == "mission_win" else config.REWARD_STRATEGIC_LOSS
                print(f"Mission ended with {'Win' if feedback == 'mission_win' else 'Loss'}. Assigning strategic reward: {strategic_reward}")
                for exp in mission_buffer:
                    # Update reward and push to main memory
                    updated_exp = (exp[0], exp[1], exp[2] + strategic_reward, exp[3])
                    replay_memory.push(updated_exp)
                mission_active = False
                mission_buffer = []
                # Trigger immediate training
                agent.train()
        elif feedback == "approve":
            reward = config.REWARD_APPROVE
            experience = (stacked_frames, action, reward, False)
            if mission_active:
                mission_buffer.append(experience)
            else:
                replay_memory.push(experience)
            # ui.perform_action(action) # TODO
        elif feedback == "reject":
            reward = config.REWARD_REJECT
            experience = (stacked_frames, action, reward, False)
            replay_memory.push(experience)
        elif feedback == "force_wait":
            wait_action_type = torch.tensor([[config.ACTION_WAIT]], device=device, dtype=torch.long)
            wait_action_params = torch.zeros(1, 4, device=device)
            wait_action = (wait_action_type, wait_action_params)
            reward = config.REWARD_APPROVE
            experience = (stacked_frames, wait_action, reward, False)
            replay_memory.push(experience)
            time.sleep(1.5)

        # 6. Train agent if needed
        if len(replay_memory) % config.TRAINING_THRESHOLD == 0 and len(replay_memory) > 0:
            print("Training...")
            agent.train()

        # time.sleep(0.1)


if __name__ == "__main__":
    main()
