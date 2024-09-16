'''
Created on 4 sept. 2024

@author: SSM9
'''

from EnvironmentEmulator import EnvironmentEmulator

if __name__ == '__main__':
    
    game = EnvironmentEmulator()
    
    game.printMap()
    
    while game.game_in_progress :
        
        
        
        print('current system state is :'+str(game.get_environment_state()))
        command = input("Let us move X ? (u/d/l/r/x)")
        
        if command == 'x':
            game.game_in_progress = False
        if command == 'u':
            game.move_up()
        elif command == 'd':
            game.move_down()
            print(str(game.reward))
        elif command == 'l':
            game.move_left()
        elif command == 'r':
            game.move_right()
        
        game.printMap()
        
    pass