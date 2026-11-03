# Message mockups: WhatsApp and SMS (task E-3)

Made by `scripts/make_message_mockups.py` from the real AirPlan code. The child
("Maya") and the forecasts are made up. **Draft wording:** not yet reviewed
by a clinician (C-8); the Spanish still needs a native-speaker review (C-7). A
third language waits for that review process (C-11).

## What the channels need

- **WhatsApp** gets the full plan. Messages a business starts must use a
  template Meta has approved, so each zone (green, yellow, red) and language
  becomes one template with the numbers as fill-in fields.
- **SMS** gets a short version and a link to the full plan. US carriers
  require opt-in and a STOP keyword, so every text ends with one.
- **SMS cost** is per part. Plain text fits 160 characters in one part (153 per
  part when longer). One character outside the GSM-7 set (like á, í, ó, ú, µ)
  switches the whole text to UCS-2: 70 characters, or 67 per part. That is
  why the SMS says "ug/m3", and why the Spanish green text says "Todo
  normal" instead of "Día normal" (3 parts down to 2).

| Language | Zone | SMS encoding | Characters | Parts billed |
|---|---|---|---|---|
| en | green | GSM-7 | 149 | 1 |
| en | yellow | GSM-7 | 259 | 2 |
| en | red | GSM-7 | 246 | 2 |
| es | green | GSM-7 | 171 | 2 |
| es | yellow | GSM-7 | 298 | 2 |
| es | red | GSM-7 | 279 | 2 |

## English

### GREEN (forecast median 7 µg/m³)

**WhatsApp**

```text
*GREEN for Maya tomorrow: normal day.*
Smoke forecast near you: about 7 µg/m³ (could reach 12). Air quality: Good.
Chance Maya has asthma symptoms tomorrow: about 16%.

- Take daily controller medicine as prescribed.
- Outdoor play is fine.
- Keep the rescue inhaler with Maya as usual.

GET HELP NOW (call 911) if Maya struggles to breathe or talk, their lips or nails turn blue or gray, or the rescue inhaler isn't helping.
Serein gives forecasts, not medical advice. Always follow the asthma action plan from your child's doctor.
```

**SMS** (GSM-7, 149 characters, 1 part)

```text
Serein: GREEN for Maya tomorrow. Smoke about 7 ug/m3 (Good). Normal day; keep the rescue inhaler as usual. Plan: serein.app/p/7Kq2 Reply STOP to end.
```

### YELLOW (forecast median 42 µg/m³)

**WhatsApp**

```text
*YELLOW for Maya tomorrow: smoke may cause symptoms.*
Smoke forecast near you: about 42 µg/m³ (could reach 70). Air quality: Unhealthy for Sensitive Groups.
Chance Maya has asthma symptoms tomorrow: about 25%.

- Keep the rescue inhaler with Maya at all times, including at school.
- Swap hard outdoor activity (running, sports practice) for indoor activity.
- Keep windows and doors closed; run an air purifier (HEPA) if you have one.
- Follow the YELLOW zone of Maya's asthma action plan if symptoms start.
- Tell the school nurse or coach that it is a smoke day.

GET HELP NOW (call 911) if Maya struggles to breathe or talk, their lips or nails turn blue or gray, or the rescue inhaler isn't helping.
Serein gives forecasts, not medical advice. Always follow the asthma action plan from your child's doctor.
```

**SMS** (GSM-7, 259 characters, 2 parts)

```text
Serein: YELLOW for Maya tomorrow. Smoke about 42 ug/m3 (Unhealthy for Sensitive Groups). Keep the rescue inhaler with Maya, choose indoor activity, close windows. Plan: serein.app/p/7Kq2 Trouble breathing or talking, or blue lips? Call 911. Reply STOP to end.
```

### RED (forecast median 160 µg/m³)

**WhatsApp**

```text
*RED for Maya tomorrow: high smoke risk. Plan to stay indoors.*
Smoke forecast near you: about 160 µg/m³ (could reach 240). Air quality: Very Unhealthy.
Chance Maya has asthma symptoms tomorrow: about 44%.

- Stay indoors in a room with windows closed and an air purifier running.
- No outdoor exercise. Use the car's recirculate setting when driving.
- Keep the rescue inhaler close and follow Maya's asthma action plan.
- Call your child's doctor today if symptoms are worse than usual.
- If you must go outside briefly, limit the time; ask your doctor whether a well-fitting mask is right for Maya.

GET HELP NOW (call 911) if Maya struggles to breathe or talk, their lips or nails turn blue or gray, or the rescue inhaler isn't helping.
Serein gives forecasts, not medical advice. Always follow the asthma action plan from your child's doctor.
```

**SMS** (GSM-7, 246 characters, 2 parts)

```text
Serein: RED for Maya tomorrow. Smoke about 160 ug/m3 (Very Unhealthy). Stay indoors with windows closed, no outdoor exercise, follow the asthma plan. Plan: serein.app/p/7Kq2 Trouble breathing or talking, or blue lips? Call 911. Reply STOP to end.
```


## Español

### GREEN (forecast median 7 µg/m³)

**WhatsApp**

```text
*VERDE para Maya mañana: día normal.*
Pronóstico de humo cerca de usted: unos 7 µg/m³ (podría llegar a 12). Índice de calidad del aire (ICA): Bueno.
Probabilidad de que Maya tenga síntomas de asma mañana: alrededor de 16%.

- Dele a Maya su medicamento de control diario como se lo recetaron.
- Puede jugar afuera.
- Que Maya lleve su inhalador de alivio rápido (de rescate) como siempre.

BUSQUE AYUDA DE INMEDIATO (llame al 911) si a Maya le cuesta respirar o hablar, se le ponen los labios o uñas azules o grises, o el inhalador de alivio rápido no le ayuda.
Serein da pronósticos, no consejos médicos. Siga siempre el plan de acción para el asma que le dio el médico de su hijo/a.
```

**SMS** (GSM-7, 171 characters, 2 parts)

```text
Serein: VERDE para Maya mañana. Humo de unos 7 ug/m3 (Bueno). Todo normal; lleve el inhalador de rescate como siempre. Plan: serein.app/p/7Kq2 Responda STOP para cancelar.
```

### YELLOW (forecast median 42 µg/m³)

**WhatsApp**

```text
*AMARILLO para Maya mañana: el humo podría causar síntomas.*
Pronóstico de humo cerca de usted: unos 42 µg/m³ (podría llegar a 70). Índice de calidad del aire (ICA): Insalubre para grupos sensibles.
Probabilidad de que Maya tenga síntomas de asma mañana: alrededor de 25%.

- Que Maya tenga su inhalador de alivio rápido (de rescate) en todo momento, también en la escuela.
- Cambie la actividad física intensa al aire libre (correr, deportes) por actividades bajo techo.
- Mantenga puertas y ventanas cerradas; use un purificador de aire portátil con filtro HEPA si tiene uno.
- Si empiezan los síntomas, siga la zona AMARILLA del plan de acción para el asma de Maya.
- Avise a la enfermera de la escuela o al entrenador que es un día de humo.

BUSQUE AYUDA DE INMEDIATO (llame al 911) si a Maya le cuesta respirar o hablar, se le ponen los labios o uñas azules o grises, o el inhalador de alivio rápido no le ayuda.
Serein da pronósticos, no consejos médicos. Siga siempre el plan de acción para el asma que le dio el médico de su hijo/a.
```

**SMS** (GSM-7, 298 characters, 2 parts)

```text
Serein: AMARILLO para Maya mañana. Humo de unos 42 ug/m3 (Insalubre para grupos sensibles). Que Maya lleve su inhalador de rescate; actividades bajo techo; ventanas cerradas. Plan: serein.app/p/7Kq2 ¿Le cuesta respirar o hablar, o tiene los labios azules? Llame al 911. Responda STOP para cancelar.
```

### RED (forecast median 160 µg/m³)

**WhatsApp**

```text
*ROJO para Maya mañana: riesgo alto por humo. Planee quedarse adentro.*
Pronóstico de humo cerca de usted: unos 160 µg/m³ (podría llegar a 240). Índice de calidad del aire (ICA): Muy insalubre.
Probabilidad de que Maya tenga síntomas de asma mañana: alrededor de 44%.

- Quédense adentro, en un cuarto con las ventanas cerradas y un purificador de aire encendido.
- Nada de ejercicio al aire libre. En el carro, use el modo de recirculación de aire.
- Tenga el inhalador de alivio rápido a la mano y siga el plan de acción para el asma de Maya.
- Llame hoy al médico de su hijo/a si los síntomas están peor que de costumbre.
- Si tiene que salir, que sea poco tiempo; pregunte a su médico si una mascarilla bien ajustada es adecuada para Maya.

BUSQUE AYUDA DE INMEDIATO (llame al 911) si a Maya le cuesta respirar o hablar, se le ponen los labios o uñas azules o grises, o el inhalador de alivio rápido no le ayuda.
Serein da pronósticos, no consejos médicos. Siga siempre el plan de acción para el asma que le dio el médico de su hijo/a.
```

**SMS** (GSM-7, 279 characters, 2 parts)

```text
Serein: ROJO para Maya mañana. Humo de unos 160 ug/m3 (Muy insalubre). Quédense adentro con ventanas cerradas, sin ejercicio afuera; sigan el plan de asma. Plan: serein.app/p/7Kq2 ¿Le cuesta respirar o hablar, o tiene los labios azules? Llame al 911. Responda STOP para cancelar.
```
