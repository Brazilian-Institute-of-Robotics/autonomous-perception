#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 19:26:01 2019

@author: nelson
"""
import math

def exponential_decay(
    learning_rate,
    global_step,
    decay_steps,
    decay_rate):

    decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
    
    return decayed_learning_rate



def polynomial_decay(
    learning_rate,
    global_step,
    decay_steps,
    end_learning_rate=0.0001,
    power=1.0,
    cycle=False):


    #If cycle is True then a multiple of decay_steps is used, the first one that is bigger than global_steps.
    if cycle:
        decay_steps = decay_steps * math.ceil(global_step / decay_steps)
    else:
        global_step = min(global_step, decay_steps)
        
    decayed_learning_rate = (learning_rate - end_learning_rate) 
    * (1 - global_step / decay_steps) ** (power) + end_learning_rate

        
    return decayed_learning_rate


def get_model_learning_rate(learning_policy,
                            base_learning_rate,
                            learning_rate_decay_step,
                            learning_rate_decay_factor,
                            training_number_of_steps,
                            learning_power,
                            slow_start_step,
                            slow_start_learning_rate,
                            slow_start_burnin_type='none'):
    
  """Gets model's learning rate.
  Computes the model's learning rate for different learning policy.
  Right now, only "step" and "poly" are supported.
  (1) The learning policy for "step" is computed as follows:
    current_learning_rate = base_learning_rate *
      learning_rate_decay_factor ^ (global_step / learning_rate_decay_step)
  See tf.train.exponential_decay for details.
  (2) The learning policy for "poly" is computed as follows:
    current_learning_rate = base_learning_rate *
      (1 - global_step / training_number_of_steps) ^ learning_power
  Args:
    learning_policy: Learning rate policy for training.
    base_learning_rate: The base learning rate for model training.
    learning_rate_decay_step: Decay the base learning rate at a fixed step.
    learning_rate_decay_factor: The rate to decay the base learning rate.
    training_number_of_steps: Number of steps for training.
    learning_power: Power used for 'poly' learning policy.
    slow_start_step: Training model with small learning rate for the first
      few steps.
    slow_start_learning_rate: The learning rate employed during slow start.
    slow_start_burnin_type: The burnin type for the slow start stage. Can be
      `none` which means no burnin or `linear` which means the learning rate
      increases linearly from slow_start_learning_rate and reaches
      base_learning_rate after slow_start_steps.
  Returns:
    Learning rate for the specified learning policy.
  Raises:
    ValueError: If learning policy or slow start burnin type is not recognized.
  """



  global_step = tf.train.get_or_create_global_step()
  adjusted_global_step = global_step

  if slow_start_burnin_type != 'none':
    adjusted_global_step -= slow_start_step

  if learning_policy == 'step':
    learning_rate = tf.train.exponential_decay(
        base_learning_rate,
        adjusted_global_step,
        learning_rate_decay_step,
        learning_rate_decay_factor,
        staircase=True)
  elif learning_policy == 'poly':
    learning_rate = tf.train.polynomial_decay(
        base_learning_rate,
        adjusted_global_step,
        training_number_of_steps,
        end_learning_rate=0,
        power=learning_power)
  else:
    raise ValueError('Unknown learning policy.')

  adjusted_slow_start_learning_rate = slow_start_learning_rate
  if slow_start_burnin_type == 'linear':
    # Do linear burnin. Increase linearly from slow_start_learning_rate and
    # reach base_learning_rate after (global_step >= slow_start_steps).
    adjusted_slow_start_learning_rate = (
        slow_start_learning_rate +
        (base_learning_rate - slow_start_learning_rate) *
        tf.to_float(global_step) / slow_start_step)
  elif slow_start_burnin_type != 'none':
    raise ValueError('Unknown burnin type.')

  # Employ small learning rate at the first few steps for warm start.
  return tf.where(global_step < slow_start_step,
                  adjusted_slow_start_learning_rate, learning_rate)