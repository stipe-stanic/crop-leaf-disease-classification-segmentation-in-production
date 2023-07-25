import time

from torch import nn, optim, Tensor

class LR_ASK:
    """Helper class that provides functionality for controlling the training process."""

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, epochs: int, ask_epoch: int):
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.ask_epoch = ask_epoch
        self.ask = True
        self.lowest_vloss = float('inf')
        self.best_weights = model.state_dict()
        self.best_epoch = 1
        self.start_time = None

    def on_train_begin(self) -> None:
        """Method called at the beginning of the training process, checks the ask_epoch and epochs values to determine
        the behavior of training
        """

        if self.ask_epoch == 0:
            print('You set ask_epoch = 0, ask_epoch will be set to 1', flush=True)
            self.ask_epoch = 1

        if self.ask_epoch >= self.epochs:
            print('Ask_epoch >= epochs, will train for', self.epochs, 'epochs', flush=True)
            self.ask = False
        elif self.epochs == 1:
            self.ask = False
        else:
            print('Training will proceed until epoch', self.ask_epoch,
                  'then you will be asked to enter H(h) to halt training or enter an integer'
                  ' to continue training for n number of epochs')

        self.start_time = time.time()

    def on_train_end(self) -> None:
        """Called at the end of the training process, loads the weights of the model with the lowest
        validation loss
        """

        print('Loading model with weights from epoch', self.best_epoch)
        self.model.load_state_dict(self.best_weights)

        train_duration = time.time() - self.start_time
        hours = int(train_duration // 3600)
        minutes = int((train_duration % 3600) // 60)
        seconds = train_duration % 60

        msg = f'Training time: {hours:02d}h:{minutes:02d}m:{seconds:02.0f}s'
        print(msg, flush=True)

    def on_epoch_end(self, epoch: int, val_loss: Tensor):
        """Called at the end of each training epoch, receives the current epoch number
        and the validation loss tensor. Saves the best weights if v_loss is lower.
        """

        # Extracts the scalar value from validation loss tensor
        v_loss = val_loss.item()
        if v_loss < self.lowest_vloss:
            self.lowest_vloss = v_loss
            self.best_weights = self.model.state_dict()
            self.best_epoch = epoch + 1

            print(f'\nValidation loss of {v_loss:.4f} is below lowest loss,'
                  f' saving weights from epoch {str(epoch + 1)} as best weights')
        else:
            print(f'\nValidation loss of {v_loss:.4f} is above lowest loss of {self.lowest_vloss:.4f}'
                  f' keeping weights from epoch {str(self.best_epoch)} as best weights')

        if self.ask and epoch + 1 == self.ask_epoch:
            print('\nEnter H(h) to end training or enter an integer for the number of additional epochs to run')
            ans = input()

            if ans == 'H' or ans == 'h':
                print(f'You entered {ans}, training halted on epoch {epoch + 1}, due to user input\n', flush=True)
                raise KeyboardInterrupt
            else:
                self.ask_epoch += int(ans)
                if self.ask_epoch > self.epochs:
                    print('\nYou specified maximum epochs as', self.epochs,
                          'cannot train for', self.ask_epoch, flush=True)
                else:
                    print(f'You entered {ans}, training will continue to epoch {self.ask_epoch}', flush=True)

                    lr = self.optimizer.param_groups[0]['lr']
                    print(f'Current LR is {lr}, enter C(c) to keep this LR or enter a float number for a new LR')

                    ans = input()
                    if ans == 'C' or ans == 'c':
                        print(f'Keeping current LR of {lr}\n')
                    else:
                        new_lr = float(ans)
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        print('Changing LR to\n', ans)
