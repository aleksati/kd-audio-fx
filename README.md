# Lightweight Neural Audio Distortion FX With Knowledge Distillation

This code repository contains scripts, code and models for the article _Compressing Neural Network Models of Audio Distortion Effects Using Knowledge Distillation Techniques_, accepted at ... conference in ... 2025.

Knowledge distillation is a technique for compressing complex and large "teacher" networks into smaller "student" networks. It offers ways to minimize the computational expenses often associated with neural networks and to optimize models for deployment and real-time usage.

In our paper, we explore the application of knowledge distillation for compressing real-time RNN models of audio distortion effects. In particular, we propose an audio-to-audio LSTM architecture for regression tasks where small audio effect networks are trained to mimic the internal representations of more extensive networks, known as feature-based knowledge distillation.

<div align="left">
 <img src="./fig/dk2.png" width="400">
</div>

https://github.com/user-attachments/assets/5847a6ff-16e5-4151-af0b-314d0ccc97b5

# Audio Examples

Our distillation architecture was evaluated on three datasets, the Blackstar HT-1 vacuum tube amplifier (HT-1), Electro-Harmonix Big Muff (Big Muff) guitar pedal, and the analog-modeled overdrive plugin DrDrive.

Below are just a few non-parametric examples comparing our distilled student models against regular students networks (non-distilled).

**64 unit networks**
DrDrive - Target - Distilled - NonDistilled
HT-1 - Target - Distilled - NonDistilled

  <audio controls>
    <source src="./examples/test.mp3" type="audio/mpeg">
    Alternate Text
  </audio>

**8 unit networks**
Big Muff - Target - Distilled - NonDistilled

# How to Run

This repository contains all the necessary utilities to use our knowledge distillation architecture. Find the code and models located inside the "./src" folder.

Clone repo.

Install the dependencies. requirements.txt

Creater starter for training the model with terminal
Same for training and inference (flag)

A starter per folder.
one teacher. not gridsearch

Another starter for conditioning. choose between conditioning and not.

https://github.com/Alec-Wright/Automated-GuitarAmpModelling/blob/main/proc_audio.py

Save mushra for later.
Send to Riccardo.

# VST Download

Avaliable soon.

aleks try with neutone
ricc with other.
